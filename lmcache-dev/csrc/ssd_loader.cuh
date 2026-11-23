// SPDX-License-Identifier: Apache-2.0
// KVPilot: layerwise load of KV from local SSD files into the paged KV cache.
#pragma once

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mem_kernels.cuh"

// One request's layerwise SSD -> pinned DRAM -> HBM load.
//
// Each (layer, chunk) value comes either from a file on local SSD (read with
// O_DIRECT by a pool of I/O threads into a pinned slot) or from pinned host
// memory that is already resident (a Mooncake DRAM replica). When a layer's
// reads are all done, the I/O thread that finished it enqueues, on the load
// stream and in layer order, one cudaMemcpyBatchAsync of all the layer's
// chunks (slot or host) into a device staging buffer and a scatter into
// kvcaches[l]. A slot is reused once its H2D is done, so nothing read from SSD
// stays in DRAM unless the request asks for a write-back (below).
//
// Write-back: start()'s `writeback[l][c]`, where nonzero, is a host buffer to
// receive the SSD value (l, c) for the caller to put back into DRAM. When the
// value's read is done, the I/O thread queues the copy for a separate pool of
// copy threads (8) and goes on reading, so
// copies delay neither that layer's H2D nor the next reads; the slot is only
// reused once its copies finish too. set_writeback() supplies the table
// later, for the reads not yet started. wait_writeback(generation()) returns
// when every copy of that start() is done.
//
// Reads run ahead of the forward, bounded by `inflight_bytes` of slots (at
// least two layers), independently of it. The number in flight starts at the
// first layer's reads and doubles per layer done, so layer 0 is not slowed by
// later layers sharing the disks. step(l) blocks the CPU only until layer l
// has been handed to the GPU, then makes the calling stream wait for layer l's
// event.
//
// Blocking pread from a thread pool rather than io_uring: the target kernel
// (5.4) has no IORING_OP_READ, and 64 threads per TP rank of ~512 KiB O_DIRECT
// reads reach 95% of the disks' sequential peak.
//
// One request at a time: start() waits for the previous request's reads to
// drain. Files stay open in a cache, so a bucket Mooncake evicts after
// open_files() remains readable.
//
// $KVPILOT_SSD_READ_GIBPS > 0 caps this process's read bandwidth: each read
// reserves its bytes on one schedule at that rate and does not complete before
// its slot ends, so the loader behaves like a disk of that bandwidth (no burst
// credit from idle time). Set it to this GPU's share of the host's SSDs (peak /
// GPUs per host) to model a fully used machine.
class SsdLayerwiseLoader {
 public:
  SsdLayerwiseLoader(int64_t num_io_threads, int64_t inflight_bytes);
  ~SsdLayerwiseLoader();
  SsdLayerwiseLoader(const SsdLayerwiseLoader&) = delete;
  SsdLayerwiseLoader& operator=(const SsdLayerwiseLoader&) = delete;

  // Opens (and caches) each file; ok[i] is false when paths[i] cannot be
  // opened, e.g. evicted since it was located.
  std::vector<bool> open_files(const std::vector<std::string>& paths);

  // file_idx[l][c] (int32 [L, C], CPU): index into `paths` of the file holding
  // chunk c's value for layer l, or -1 for a host chunk. offsets[l][c] (int64
  // [L, C], CPU): the value's byte offset in its file, or the host address of
  // a host chunk. sizes[c]: bytes of chunk c to load, token-major; chunks are
  // packed back to back into the staging buffers.
  // writeback: optional int64 [L, C] CPU tensor of host addresses, 0 where an
  // SSD value is not written back; ignored for host chunks.
  // evict_events: as for LayerwiseLoader (a LayerEvictor's events()).
  std::shared_ptr<LoadTiming> start(
      const std::vector<std::string>& paths, const torch::Tensor& file_idx,
      const torch::Tensor& offsets, const std::vector<int64_t>& sizes,
      torch::Tensor staging0, torch::Tensor staging1,
      std::vector<torch::Tensor> kvcaches, torch::Tensor slot_mapping,
      const GPUKVFormat gpu_kv_format, const bool timing,
      const std::optional<torch::Tensor>& writeback,
      const std::vector<int64_t>& evict_events);
  void step(const int64_t layer, const bool wait);
  // Number of the latest start(); pass it to set_writeback / wait_writeback.
  int64_t generation();
  // The write-back table of start() number `gen`, arriving after it: arms
  // the values whose reads have not started. Returns uint8 [L, C], 1 where a
  // value was armed (all 0 if that load is over or failed).
  torch::Tensor set_writeback(const int64_t gen, const torch::Tensor& writeback);
  // Blocks until every write-back copy of start() number `gen` is done (or
  // a later start() has begun, which implies it). False if that load failed.
  bool wait_writeback(const int64_t gen);

 private:
  struct Read {
    int64_t layer;
    int64_t chunk;
    int fd;
    int64_t aligned_off;  // file offset rounded down to 4 KiB
    int64_t aligned_len;  // [aligned_off, value end) rounded up to 4 KiB
    int64_t slot_off;     // where the aligned read lands in the slot
    int64_t need;         // bytes the read must return: skip + size
    char* wb_dst = nullptr;  // write-back buffer for the value, if any
    int64_t wb_len = 0;      // value bytes, at slot_off + need - wb_len
  };

  struct Copy {
    char* dst;
    const char* src;
    int64_t len;
    int64_t slot;
  };

  void io_main();
  void copy_main();
  bool claimable_locked();
  void submit_ready_locked();
  void note_writeback_done_locked();
  int open_cached_locked(const std::string& path);
  // End of this read's slot on the bandwidth schedule.
  std::chrono::steady_clock::time_point reserve_bandwidth(int64_t bytes);

  const int64_t num_threads_, inflight_bytes_;
  int device_ = -1;
  char* arena_ = nullptr;  // pinned, page aligned; num_slots_ x slot_bytes_
  int64_t arena_bytes_ = 0, slot_bytes_ = 0, num_slots_ = 0;
  std::vector<cudaEvent_t> slot_free_;  // per slot: H2D out of it finished
  std::vector<cudaEvent_t> done_;       // per layer: scatter finished
  std::unordered_map<std::string, int> fds_;

  // This request's plan: reads (layer-major) and, per layer and chunk, the
  // host address the H2D copies from (inside a slot for SSD chunks).
  std::vector<Read> reads_;
  std::vector<std::vector<int64_t>> src_;  // [L][C]; slot chunks: slot offset
  std::vector<std::vector<bool>> in_slot_;
  std::vector<int64_t> sizes_;

  torch::Tensor staging_[2];
  std::vector<torch::Tensor> kvcaches_;
  torch::Tensor slots_map_;
  std::vector<cudaEvent_t> evict_events_;
  GPUKVFormat fmt_ = GPUKVFormat::NL_X_TWO_NB_BS_NH_HS;
  int64_t num_layers_ = 0, num_tokens_ = 0;
  std::optional<c10::cuda::CUDAStream> stream_;
  std::shared_ptr<LoadTiming> timing_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::thread> threads_;
  size_t next_read_ = 0;       // next read to claim
  int64_t inflight_ = 0;       // reads claimed and not finished
  int64_t ramp_base_ = 1;      // reads of the first layer that has any
  std::vector<int64_t> left_;  // per layer: reads not yet done
  int64_t submitted_ = 0;      // layers handed to the GPU, in order
  bool stop_ = false;
  std::string error_;
  int64_t copies_ = 0;             // write-back copies queued or running
  std::vector<int64_t> copying_;   // per slot: write-back copies out of it
  std::deque<Copy> copy_q_;        // write-back copies not started
  std::condition_variable copy_cv_;
  std::vector<std::thread> copy_threads_;
  int64_t gen_ = 0;                // start() count
  int64_t wb_done_gen_ = 0;        // latest start() whose copies all finished
  int64_t wb_failed_gen_ = 0;      // latest start() whose load failed

  double ns_per_byte_ = 0.0;  // 0: no bandwidth cap
  std::mutex rate_mu_;
  std::chrono::steady_clock::time_point rate_next_{};
};
