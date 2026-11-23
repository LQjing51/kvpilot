// SPDX-License-Identifier: Apache-2.0
#include "ssd_loader.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

namespace {
constexpr int64_t kAlign = 4096;
// Bucket ids never repeat, so the cache only grows; past this many files it is
// dropped while idle. Kept under the usual 64 Ki open-file limit.
constexpr size_t kMaxCachedFds = 16384;
int64_t align_down(int64_t x) { return x & ~(kAlign - 1); }
int64_t align_up(int64_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }
}  // namespace

SsdLayerwiseLoader::SsdLayerwiseLoader(int64_t num_io_threads,
                                       int64_t inflight_bytes)
    : num_threads_(std::max<int64_t>(1, num_io_threads)),
      inflight_bytes_(std::max<int64_t>(kAlign, inflight_bytes)) {
  if (const char* v = std::getenv("KVPILOT_SSD_READ_GIBPS")) {
    const double gibps = std::atof(v);
    if (gibps > 0) ns_per_byte_ = 1e9 / (gibps * double(1ull << 30));
  }
  // Pinning the arena and starting the threads take ~80 ms; done here, the
  // first SSD load does not pay for them.
  cudaGetDevice(&device_);
  void* p = nullptr;
  if (cudaHostAlloc(&p, inflight_bytes_, cudaHostAllocDefault) == cudaSuccess) {
    arena_ = static_cast<char*>(p);
    arena_bytes_ = inflight_bytes_;
  }
  for (int64_t i = 0; i < num_threads_; ++i) {
    threads_.emplace_back(&SsdLayerwiseLoader::io_main, this);
  }
  // Write-back copies (slot -> DRAM replica) run on their own threads so they
  // never delay a layer's H2D or take a read's place.
  constexpr int64_t kCopyThreads = 8;
  for (int64_t i = 0; i < kCopyThreads; ++i) {
    copy_threads_.emplace_back(&SsdLayerwiseLoader::copy_main, this);
  }
}

SsdLayerwiseLoader::~SsdLayerwiseLoader() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  copy_cv_.notify_all();
  for (auto& t : threads_) t.join();
  for (auto& t : copy_threads_) t.join();
  for (cudaEvent_t e : slot_free_) cudaEventSynchronize(e);
  if (arena_) cudaFreeHost(arena_);
  for (cudaEvent_t e : slot_free_) cudaEventDestroy(e);
  for (cudaEvent_t e : done_) cudaEventDestroy(e);
  for (auto& [_, fd] : fds_) close(fd);
}

int SsdLayerwiseLoader::open_cached_locked(const std::string& path) {
  auto it = fds_.find(path);
  if (it != fds_.end()) return it->second;
  const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd >= 0) fds_.emplace(path, fd);
  return fd;
}

std::vector<bool> SsdLayerwiseLoader::open_files(
    const std::vector<std::string>& paths) {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<bool> ok(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    ok[i] = open_cached_locked(paths[i]) >= 0;
  }
  return ok;
}

std::shared_ptr<LoadTiming> SsdLayerwiseLoader::start(
    const std::vector<std::string>& paths, const torch::Tensor& file_idx,
    const torch::Tensor& offsets, const std::vector<int64_t>& sizes,
    torch::Tensor staging0, torch::Tensor staging1,
    std::vector<torch::Tensor> kvcaches, torch::Tensor slot_mapping,
    const GPUKVFormat gpu_kv_format, const bool timing,
    const std::optional<torch::Tensor>& writeback,
    const std::vector<int64_t>& evict_events) {
  TORCH_CHECK(!file_idx.is_cuda() && file_idx.scalar_type() == torch::kInt32 &&
                  file_idx.dim() == 2 && file_idx.is_contiguous(),
              "file_idx must be a contiguous [layers, chunks] int32 CPU tensor");
  TORCH_CHECK(!offsets.is_cuda() && offsets.scalar_type() == torch::kInt64 &&
                  offsets.sizes() == file_idx.sizes() &&
                  offsets.is_contiguous(),
              "offsets must be a contiguous int64 CPU tensor shaped like "
              "file_idx");
  TORCH_CHECK(!writeback || (!writeback->is_cuda() &&
                             writeback->scalar_type() == torch::kInt64 &&
                             writeback->sizes() == file_idx.sizes() &&
                             writeback->is_contiguous()),
              "writeback must be a contiguous int64 CPU tensor shaped like "
              "file_idx");
  const int64_t num_layers = file_idx.size(0), num_chunks = file_idx.size(1);
  TORCH_CHECK(num_layers == static_cast<int64_t>(kvcaches.size()) &&
                  num_chunks == static_cast<int64_t>(sizes.size()) &&
                  num_chunks > 0,
              "SsdLayerwiseLoader: plan does not match kvcaches / sizes");
  TORCH_CHECK(evict_events.empty() ||
                  static_cast<int64_t>(evict_events.size()) >= num_layers,
              "SsdLayerwiseLoader: evict_events must have one event per layer");
  int64_t total = 0;
  for (int64_t s : sizes) total += s;
  TORCH_CHECK(total <= static_cast<int64_t>(staging0.nbytes()) &&
                  total <= static_cast<int64_t>(staging1.nbytes()),
              "SsdLayerwiseLoader: staging buffer too small");

  std::unique_lock<std::mutex> lk(mu_);
  // The previous request's reads and write-back copies must be finished
  // before its plan and slots are replaced. Its H2Ds out of the slots may
  // still be on the GPU; the slot-free events cover those below. Nothing
  // below touches the members the I/O threads read until every check has
  // passed.
  cv_.wait(lk, [&] {
    return inflight_ == 0 && copies_ == 0 &&
           (submitted_ == num_layers_ || !error_.empty());
  });

  const c10::cuda::CUDAGuard guard(staging0.device());
  TORCH_CHECK(staging0.device().index() == device_,
              "SsdLayerwiseLoader: created on device ", device_,
              ", used on device ", staging0.device().index());
  if (fds_.size() > kMaxCachedFds) {
    for (auto& [_, fd] : fds_) close(fd);
    fds_.clear();
  }
  std::vector<int> fds(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    fds[i] = open_cached_locked(paths[i]);
    TORCH_CHECK(fds[i] >= 0, "SsdLayerwiseLoader: open ", paths[i], ": ",
                std::strerror(errno));
  }

  // Plan: per layer, SSD chunks get consecutive aligned ranges of the slot.
  const int32_t* fidx = file_idx.data_ptr<int32_t>();
  const int64_t* off = offsets.data_ptr<int64_t>();
  const int64_t* wb = writeback ? writeback->data_ptr<int64_t>() : nullptr;
  std::vector<Read> reads;
  std::vector<std::vector<int64_t>> src(num_layers,
                                        std::vector<int64_t>(num_chunks));
  std::vector<std::vector<bool>> in_slot(num_layers,
                                         std::vector<bool>(num_chunks));
  std::vector<int64_t> left(num_layers, 0);
  int64_t need = 0;
  for (int64_t l = 0; l < num_layers; ++l) {
    int64_t slot_off = 0;
    for (int64_t c = 0; c < num_chunks; ++c) {
      const int32_t f = fidx[l * num_chunks + c];
      const int64_t o = off[l * num_chunks + c];
      if (f < 0) {
        src[l][c] = o;
        continue;
      }
      TORCH_CHECK(f < static_cast<int32_t>(fds.size()),
                  "SsdLayerwiseLoader: bad file index");
      Read r;
      r.layer = l;
      r.chunk = c;
      r.fd = fds[f];
      r.aligned_off = align_down(o);
      r.need = o - r.aligned_off + sizes[c];
      r.aligned_len = align_up(r.need);
      r.slot_off = slot_off;
      if (wb && wb[l * num_chunks + c]) {
        r.wb_dst = reinterpret_cast<char*>(wb[l * num_chunks + c]);
        r.wb_len = sizes[c];
      }
      src[l][c] = slot_off + (o - r.aligned_off);
      in_slot[l][c] = true;
      slot_off += r.aligned_len;
      reads.push_back(r);
      ++left[l];
    }
    need = std::max(need, slot_off);
  }

  while (static_cast<int64_t>(slot_free_.size()) < num_layers) {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    slot_free_.push_back(e);
  }
  while (static_cast<int64_t>(done_.size()) < num_layers) {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    done_.push_back(e);
  }
  // The slot layout changes per request, so every earlier H2D out of the
  // arena must have finished; normally it has, long before.
  for (cudaEvent_t e : slot_free_) cudaEventSynchronize(e);
  // Slots: as many layers as fit in inflight_bytes_, at least two.
  const int64_t slot_bytes = std::max<int64_t>(need, kAlign);
  const int64_t num_slots = std::min<int64_t>(
      num_layers, std::max<int64_t>(2, inflight_bytes_ / slot_bytes));
  if (num_slots * slot_bytes > arena_bytes_) {
    if (arena_) cudaFreeHost(arena_);
    arena_ = nullptr;
    arena_bytes_ = 0;
    const int64_t bytes = std::max(num_slots * slot_bytes, inflight_bytes_);
    void* p = nullptr;
    const cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    TORCH_CHECK(err == cudaSuccess, "SsdLayerwiseLoader: cudaHostAlloc of ",
                bytes, " bytes failed: ", cudaGetErrorString(err));
    arena_ = static_cast<char*>(p);
    arena_bytes_ = bytes;
  }

  ramp_base_ = 1;
  for (int64_t n : left) {
    if (n > 0) {
      ramp_base_ = n;
      break;
    }
  }
  ++gen_;
  reads_ = std::move(reads);
  src_ = std::move(src);
  in_slot_ = std::move(in_slot);
  left_ = std::move(left);
  slot_bytes_ = slot_bytes;
  num_slots_ = num_slots;
  copying_.assign(num_slots, 0);
  staging_[0] = std::move(staging0);
  staging_[1] = std::move(staging1);
  kvcaches_ = std::move(kvcaches);
  slots_map_ = std::move(slot_mapping);
  evict_events_.clear();
  for (int64_t e : evict_events) evict_events_.push_back(reinterpret_cast<cudaEvent_t>(e));
  num_tokens_ = slots_map_.size(0);
  sizes_ = sizes;
  fmt_ = gpu_kv_format;
  num_layers_ = num_layers;
  next_read_ = 0;
  submitted_ = 0;
  error_.clear();
  stream_ = at::cuda::getCurrentCUDAStream();
  timing_ = timing ? std::make_shared<LoadTiming>() : nullptr;
  if (timing_) {
    // Span starts before any read, so it covers SSD time, not just H2D.
    cudaEventCreate(&timing_->t0);
    cudaEventRecord(timing_->t0, stream_->stream());
  }
  // Layers with no SSD chunk at the front are ready now.
  submit_ready_locked();
  lk.unlock();
  cv_.notify_all();
  return timing_;
}

bool SsdLayerwiseLoader::claimable_locked() {
  if (!error_.empty() || next_read_ >= reads_.size()) return false;
  // Reads in flight share the disks' bandwidth, so with every thread busy
  // from the start layer 0 lands only when the first ~num_threads reads do,
  // and the forward waits that long. Start from the first layer's reads and
  // double per layer handed to the GPU.
  const int64_t cap =
      std::min<int64_t>(num_threads_, ramp_base_ << std::min<int64_t>(submitted_, 20));
  if (inflight_ >= cap) return false;
  const int64_t layer = reads_[next_read_].layer;
  if (layer < num_slots_) return true;  // start() waited for every slot
  // Slot layer % S last held layer - S of this request: its H2D must be
  // enqueued and finished, and its write-back copies done.
  if (submitted_ <= layer - num_slots_) return false;
  if (copying_[layer % num_slots_] != 0) return false;
  return cudaEventQuery(slot_free_[layer % num_slots_]) == cudaSuccess;
}

void SsdLayerwiseLoader::note_writeback_done_locked() {
  if (error_.empty() && copies_ == 0 && submitted_ == num_layers_) {
    wb_done_gen_ = gen_;
  }
}

void SsdLayerwiseLoader::submit_ready_locked() {
  // Hand finished layers to the GPU strictly in order: staging buffers
  // alternate by layer and rely on load-stream order for reuse.
  if (!error_.empty() || submitted_ >= num_layers_ || left_[submitted_] != 0)
    return;
  const c10::cuda::CUDAStreamGuard guard(*stream_);
  const cudaStream_t s = stream_->stream();
  const int64_t num_chunks = static_cast<int64_t>(sizes_.size());
  std::vector<int64_t> src(num_chunks);
  while (submitted_ < num_layers_ && left_[submitted_] == 0) {
    const int64_t l = submitted_;
    const int64_t slot = l % num_slots_;
    char* base = arena_ + slot * slot_bytes_;
    bool uses_slot = false;
    for (int64_t c = 0; c < num_chunks; ++c) {
      src[c] = in_slot_[l][c] ? reinterpret_cast<int64_t>(base + src_[l][c])
                              : src_[l][c];
      uses_slot |= in_slot_[l][c];
    }
    if (timing_) {
      cudaEvent_t e;
      cudaEventCreate(&e);
      cudaEventRecord(e, s);
      timing_->begin.push_back(e);
    }
    torch::Tensor& buf = staging_[l % 2];
    batched_memcpy_h2d(src, sizes_, buf);
    if (uses_slot) cudaEventRecord(slot_free_[slot], s);
    // The scatter overwrites lent blocks: their old KV must be out first.
    if (!evict_events_.empty()) cudaStreamWaitEvent(s, evict_events_[l], 0);
    torch::Tensor dst = buf.narrow(0, 0, num_tokens_);
    single_layer_kv_transfer(dst, kvcaches_[l], slots_map_,
                             TransferDirection::H2D, fmt_, true);
    cudaEventRecord(done_[l], s);
    if (timing_) {
      cudaEvent_t e;
      cudaEventCreate(&e);
      cudaEventRecord(e, s);
      timing_->end.push_back(e);
    }
    ++submitted_;
  }
  note_writeback_done_locked();
}

void SsdLayerwiseLoader::io_main() {
  cudaSetDevice(device_);
  std::unique_lock<std::mutex> lk(mu_);
  while (!stop_) {
    if (!error_.empty() || next_read_ >= reads_.size()) {
      cv_.wait(lk);  // idle until the next start()
      continue;
    }
    if (!claimable_locked()) {
      // Blocked on a slot whose H2D is still on the GPU; that event is not
      // signalled through cv_, so re-poll. Not tighter: every poll is a
      // cudaEventQuery, which shares driver locks with the forward's launches.
      cv_.wait_for(lk, std::chrono::microseconds(200));
      continue;
    }
    const Read r = reads_[next_read_++];
    char* dst = arena_ + (r.layer % num_slots_) * slot_bytes_ + r.slot_off;
    ++inflight_;
    lk.unlock();
    std::optional<std::chrono::steady_clock::time_point> slot_end;
    if (ns_per_byte_ > 0) slot_end = reserve_bandwidth(r.aligned_len);

    // O_DIRECT: offset, length and buffer are 4 KiB aligned. A read may come
    // back short only at end of file, where the value must still be whole.
    int64_t got = 0;
    std::string err;
    while (got < r.need) {
      const ssize_t n =
          pread(r.fd, dst + got, r.aligned_len - got, r.aligned_off + got);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) {
        err = std::strerror(errno);
        break;
      }
      if (n == 0) {
        err = "unexpected end of file";
        break;
      }
      got += n;
      if (got < r.need && got % kAlign != 0) {
        err = "short read";
        break;
      }
    }
    if (slot_end) std::this_thread::sleep_until(*slot_end);

    lk.lock();
    --inflight_;
    if (!err.empty()) {
      if (error_.empty()) error_ = "SsdLayerwiseLoader: read failed: " + err;
      wb_failed_gen_ = gen_;
    } else {
      if (r.wb_dst) {
        // A copy thread takes it; copying_ keeps the slot from being refilled.
        const int64_t slot = r.layer % num_slots_;
        ++copies_;
        ++copying_[slot];
        copy_q_.push_back({r.wb_dst, dst + (r.need - r.wb_len), r.wb_len, slot});
        copy_cv_.notify_one();
      }
      if (--left_[r.layer] == 0) submit_ready_locked();
    }
    cv_.notify_all();
  }
}

void SsdLayerwiseLoader::copy_main() {
  std::unique_lock<std::mutex> lk(mu_);
  while (true) {
    copy_cv_.wait(lk, [&] { return stop_ || !copy_q_.empty(); });
    if (copy_q_.empty()) break;  // stop_, nothing left to copy
    const Copy c = copy_q_.front();
    copy_q_.pop_front();
    lk.unlock();
    std::memcpy(c.dst, c.src, c.len);
    lk.lock();
    --copies_;
    --copying_[c.slot];
    note_writeback_done_locked();
    cv_.notify_all();
  }
}

int64_t SsdLayerwiseLoader::generation() {
  std::lock_guard<std::mutex> lk(mu_);
  return gen_;
}

torch::Tensor SsdLayerwiseLoader::set_writeback(const int64_t gen,
                                                const torch::Tensor& writeback) {
  TORCH_CHECK(!writeback.is_cuda() && writeback.scalar_type() == torch::kInt64 &&
                  writeback.dim() == 2 && writeback.is_contiguous(),
              "writeback must be a contiguous [layers, chunks] int64 CPU tensor");
  std::lock_guard<std::mutex> lk(mu_);
  const int64_t num_chunks = static_cast<int64_t>(sizes_.size());
  torch::Tensor armed = torch::zeros({num_layers_, num_chunks}, torch::kUInt8);
  if (gen != gen_ || !error_.empty() || writeback.size(0) != num_layers_ ||
      writeback.size(1) != num_chunks) {
    return armed;
  }
  const int64_t* wb = writeback.data_ptr<int64_t>();
  uint8_t* out = armed.data_ptr<uint8_t>();
  // Reads from next_read_ on have not been claimed: an I/O thread copies
  // the Read under mu_ when it claims one, so it sees the destination.
  for (size_t i = next_read_; i < reads_.size(); ++i) {
    Read& r = reads_[i];
    const int64_t at = r.layer * num_chunks + r.chunk;
    if (wb[at] && !r.wb_dst) {
      r.wb_dst = reinterpret_cast<char*>(wb[at]);
      r.wb_len = sizes_[r.chunk];
      out[at] = 1;
    }
  }
  return armed;
}

bool SsdLayerwiseLoader::wait_writeback(const int64_t gen) {
  std::unique_lock<std::mutex> lk(mu_);
  // A later start() waited for this one's copies; a failed load claims no
  // more reads, so it is over once its running copies finish.
  cv_.wait(lk, [&] {
    return stop_ || gen_ > gen || wb_done_gen_ >= gen ||
           (wb_failed_gen_ >= gen && copies_ == 0);
  });
  return wb_done_gen_ >= gen && wb_failed_gen_ < gen;
}

std::chrono::steady_clock::time_point SsdLayerwiseLoader::reserve_bandwidth(
    const int64_t bytes) {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(rate_mu_);
  if (rate_next_ < now) rate_next_ = now;
  rate_next_ += std::chrono::nanoseconds(
      static_cast<int64_t>(static_cast<double>(bytes) * ns_per_byte_));
  return rate_next_;
}

void SsdLayerwiseLoader::step(const int64_t layer, const bool wait) {
  {
    std::unique_lock<std::mutex> lk(mu_);
    TORCH_CHECK(layer >= 0 && layer < num_layers_,
                "SsdLayerwiseLoader::step: bad layer");
    cv_.wait(lk, [&] { return submitted_ > layer || !error_.empty(); });
    TORCH_CHECK(error_.empty(), error_);
  }
  if (wait) {
    TORCH_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(),
                                    done_[layer], 0) == cudaSuccess,
                "cudaStreamWaitEvent failed");
  }
}
