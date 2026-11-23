// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)

The logic for identifying the format currently lives in
`lmcache/v1/gpu_connector/utils.py`
*/
enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  /*
  used by:
  - vLLM CROSS_LAYER mode
  */

  NL_X_TWO_NB_BS_NH_HS = 1,
  /*
  used by:
  - vLLM non-MLA flash attention
  */

  NL_X_NB_TWO_BS_NH_HS = 2,
  /*
  used by:
  - vLLM non-MLA flash infer
  */

  NL_X_NB_BS_HS = 3,
  /*
  used by:
  - vLLM MLA
  */

  TWO_X_NL_X_NBBS_NH_HS = 4,
  /*
  used by:
  - SGLang MHA (flash attention and flash infer)
  */

  NL_X_NBBS_ONE_HS = 5,
  /*
  used by:
  - SGLang MLA
  */

  NL_X_TWO_NB_NH_BS_HS = 6,
  /*
  used by:
  - vLLM non-MLA flash attention (HND layout)
  physical shape per layer: [2, num_blocks, num_heads, block_size, head_size]
  */

  NL_X_NB_TWO_NH_BS_HS = 7,
  /*
  used by:
  - vLLM non-MLA flash infer (HND layout)
  physical shape per layer: [num_blocks, 2, num_heads, block_size, head_size]
  */

  NB_NL_TWO_NH_BS_HS = 8,
  /*
  used by:
  - TRT-LLM cross-layer (HND layout)
  physical shape: [num_blocks, num_layers, 2, num_heads, block_size, head_size]
  */

  TWO_X_NL_X_NB_BS_NH_HS = 9,
  /*
  used by:
  - SGLang MHA via the MP daemon path
  physical shape per layer: [num_blocks, block_size, num_heads, head_size]
  */
};

void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format, const int block_size = 0,
    const int head_size = 0, const int skip_prefix_n_tokens = 0);

// collapses to multi_layer_kv_transfer for MLA
void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const GPUKVFormat gpu_kv_format,
                              const bool token_major = false);

void multi_chunk_kv_transfer_h2d(const torch::Tensor& chunk_ptrs,
                                 torch::Tensor& vllm_key_value_cache,
                                 const torch::Tensor& slot_mapping,
                                 const int chunk_size,
                                 const GPUKVFormat gpu_kv_format);

void batched_memcpy_h2d(const std::vector<int64_t>& src_ptrs,
                        const std::vector<int64_t>& sizes, torch::Tensor& dst);

// KVPilot: the reverse of batched_memcpy_h2d. Values packed back to back in
// device tensor src go to host addresses dst_ptrs (pinned or registered, e.g.
// Mooncake replicas) in one cudaMemcpyBatchAsync on the current stream, which
// must not be the legacy default stream (cudaMemcpyBatchAsync rejects it); a 0
// address skips its value, which still takes its place in src.
void batched_memcpy_d2h(torch::Tensor& src, const std::vector<int64_t>& dst_ptrs,
                        const std::vector<int64_t>& sizes);

// KVPilot exclusive unblocking: HBM blocks lent to a request's load still hold
// KV whose only copy they are; it goes to its DRAM replicas before the load
// overwrites them. start() enqueues every layer on the current stream (not the
// load stream): gather the blocks' slots into a staging buffer (two,
// alternating by layer), one cudaMemcpyBatchAsync into the replicas, then an
// event. A loader given events() makes its scatter into layer l wait for event
// l, so the D2H of a layer runs alongside the load's H2D (the other PCIe
// direction) and only the scatter can wait. Events are reused across requests
// like LayerwiseLoader's.
class LayerEvictor {
 public:
  LayerEvictor() = default;
  ~LayerEvictor();
  LayerEvictor(const LayerEvictor&) = delete;
  LayerEvictor& operator=(const LayerEvictor&) = delete;
  // layer_ptrs: contiguous int64 [layers, blocks] CPU tensor, the replica of
  // each block's value per layer; slot_mapping: the blocks' slots, block after
  // block.
  void start(const torch::Tensor& layer_ptrs, torch::Tensor staging0,
             torch::Tensor staging1, std::vector<torch::Tensor> kvcaches,
             torch::Tensor slot_mapping, const GPUKVFormat gpu_kv_format);
  // The per-layer events, as cudaEvent_t handles.
  std::vector<int64_t> events() const;

 private:
  std::vector<cudaEvent_t> done_;
  torch::Tensor staging_[2];
  std::vector<torch::Tensor> kvcaches_;
  torch::Tensor slots_;
};

// KVPilot: per-layer timing of one layerwise load, recorded only when asked
// for. result() blocks until the last layer is done and returns (summed
// per-layer copy + scatter ms, span ms, layer 0 ready ms). The span and the
// layer-0 time run from t0 when set (the SSD loader records it before any
// read) else from the first layer's begin.
struct LoadTiming {
  std::vector<cudaEvent_t> begin, end;
  cudaEvent_t t0 = nullptr;
  ~LoadTiming();
  std::tuple<double, double, double> result();
};

// KVPilot: one request's layerwise H2D from pinned host chunks into the paged
// KV. Layer l is one cudaMemcpyBatchAsync of its chunks (adjacent host ranges
// merged) into a staging buffer (two, alternating by layer), a scatter into
// kvcaches[l], then an event. start() runs on the load stream and enqueues the
// first `lookahead` layers (pass num_layers to queue every layer up front);
// step(l) enqueues layer l + lookahead if any remain, then makes the calling
// stream wait for layer l. All per-layer work happens here with the GIL
// released, so the forward thread pays one call per layer. Events are reused
// across requests: a wait binds to the record preceding it, so a later request
// re-recording an event does not move earlier waits. evict_events: a
// LayerEvictor's events() when the load overwrites lent blocks, else empty.
class LayerwiseLoader {
 public:
  LayerwiseLoader() = default;
  ~LayerwiseLoader();
  LayerwiseLoader(const LayerwiseLoader&) = delete;
  LayerwiseLoader& operator=(const LayerwiseLoader&) = delete;
  std::shared_ptr<LoadTiming> start(const torch::Tensor& layer_ptrs,
                                    const std::vector<int64_t>& sizes,
                                    torch::Tensor staging0, torch::Tensor staging1,
                                    std::vector<torch::Tensor> kvcaches,
                                    torch::Tensor slot_mapping,
                                    const GPUKVFormat gpu_kv_format,
                                    const int64_t lookahead, const bool timing,
                                    const std::vector<int64_t>& evict_events);
  void step(const int64_t layer, const bool wait);

 private:
  void enqueue(const int64_t layer);

  std::vector<cudaEvent_t> done_;
  std::vector<std::vector<void*>> dsts_;
  std::vector<std::vector<const void*>> srcs_;
  std::vector<std::vector<size_t>> bytes_;
  torch::Tensor staging_[2];
  std::vector<torch::Tensor> kvcaches_;
  torch::Tensor slots_;
  std::vector<cudaEvent_t> evict_events_;
  GPUKVFormat fmt_ = GPUKVFormat::NL_X_TWO_NB_BS_NH_HS;
  int64_t num_layers_ = 0;
  int64_t num_tokens_ = 0;
  int64_t lookahead_ = 0;
  std::optional<c10::cuda::CUDAStream> stream_;
  std::shared_ptr<LoadTiming> timing_;
};


void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const TransferDirection direction,
                                  const bool token_major = false);

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);
