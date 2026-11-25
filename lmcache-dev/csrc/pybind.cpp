// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mem_kernels.cuh"
#include "ssd_loader.cuh"
#include "mp_mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include "event_recorder.h"
#include "completion_recorder.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<GPUKVFormat>(m, "GPUKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", GPUKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", GPUKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", GPUKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", GPUKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", GPUKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", GPUKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", GPUKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", GPUKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", GPUKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .export_values();
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("block_size") = 0,
        py::arg("head_size") = 0, py::arg("skip_prefix_n_tokens") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"), py::arg("gpu_kv_format"),
        py::arg("token_major") = false);
  m.def("multi_chunk_kv_transfer_h2d", &multi_chunk_kv_transfer_h2d,
        py::arg("chunk_ptrs"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("chunk_size"),
        py::arg("gpu_kv_format"), py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy_h2d", &batched_memcpy_h2d, py::arg("src_ptrs"),
        py::arg("sizes"), py::arg("dst"),
        py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy_d2h", &batched_memcpy_d2h, py::arg("src"),
        py::arg("dst_ptrs"), py::arg("sizes"),
        py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy_async", &batched_memcpy_async, py::arg("dst_ptrs"),
        py::arg("src_ptrs"), py::arg("sizes"),
        py::call_guard<py::gil_scoped_release>());
  py::class_<LoadTiming, std::shared_ptr<LoadTiming>>(m, "LoadTiming")
      .def("result", &LoadTiming::result,
           py::call_guard<py::gil_scoped_release>());
  py::class_<LayerEvictor>(m, "LayerEvictor")
      .def(py::init<>())
      .def("start", &LayerEvictor::start, py::arg("layer_ptrs"),
           py::arg("staging0"), py::arg("staging1"), py::arg("kvcaches"),
           py::arg("slot_mapping"), py::arg("gpu_kv_format"),
           py::arg("side_src") = py::none(), py::arg("side_dst") = py::none(),
           py::arg("side_bytes") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("events", &LayerEvictor::events);
  py::class_<LayerwiseLoader>(m, "LayerwiseLoader")
      .def(py::init<>())
      .def("start", &LayerwiseLoader::start, py::arg("layer_ptrs"),
           py::arg("sizes"), py::arg("staging0"), py::arg("staging1"),
           py::arg("kvcaches"), py::arg("slot_mapping"), py::arg("gpu_kv_format"),
           py::arg("lookahead"), py::arg("timing"),
           py::arg("evict_events") = std::vector<int64_t>(),
           py::arg("side_dst") = py::none(), py::arg("side_bytes") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("step", &LayerwiseLoader::step, py::arg("layer"), py::arg("wait"),
           py::call_guard<py::gil_scoped_release>());
  py::class_<SsdLayerwiseLoader>(m, "SsdLayerwiseLoader")
      .def(py::init<int64_t, int64_t>(), py::arg("num_io_threads"),
           py::arg("inflight_bytes"))
      .def("open_files", &SsdLayerwiseLoader::open_files, py::arg("paths"),
           py::call_guard<py::gil_scoped_release>())
      .def("start", &SsdLayerwiseLoader::start, py::arg("paths"),
           py::arg("file_idx"), py::arg("offsets"), py::arg("sizes"),
           py::arg("staging0"), py::arg("staging1"), py::arg("kvcaches"),
           py::arg("slot_mapping"), py::arg("gpu_kv_format"), py::arg("timing"),
           py::arg("writeback") = py::none(),
           py::arg("evict_events") = std::vector<int64_t>(),
           py::arg("side_dst") = py::none(), py::arg("side_bytes") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("step", &SsdLayerwiseLoader::step, py::arg("layer"),
           py::arg("wait"), py::call_guard<py::gil_scoped_release>())
      .def("generation", &SsdLayerwiseLoader::generation)
      .def("set_writeback", &SsdLayerwiseLoader::set_writeback, py::arg("gen"),
           py::arg("writeback"), py::call_guard<py::gil_scoped_release>())
      .def("wait_writeback", &SsdLayerwiseLoader::wait_writeback,
           py::arg("gen"), py::call_guard<py::gil_scoped_release>());
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_hugepage_pinned_ptr", &alloc_hugepage_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_ptr", &free_hugepage_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_hugepage_pinned_numa_ptr", &alloc_hugepage_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_numa_ptr", &free_hugepage_pinned_numa_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("alloc_shm_pinned_ptr", &alloc_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_shm_pinned_ptr", &free_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy", &batched_memcpy, py::arg("src_ptrs"),
        py::arg("dst_ptrs"), py::arg("sizes"),
        py::call_guard<py::gil_scoped_release>());
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
  m.def("multi_layer_block_kv_transfer", &multi_layer_block_kv_transfer,
        py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
        py::arg("block_ids"), py::arg("device"), py::arg("direction"),
        py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
        py::arg("gpu_kv_format"), py::arg("skip_prefix_n_blocks"),
        py::call_guard<py::gil_scoped_release>());
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc")
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems);
  m.def("record_event_on_stream", &record_event_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("event_type_name"),
        py::arg("session_id"), py::arg("str_metadata"), py::arg("int_metadata"),
        py::call_guard<py::gil_scoped_release>());
  m.def("drain_recorded_events", &drain_recorded_events);
  m.def("record_completion_on_stream", &record_completion_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("kind"), py::arg("payload"),
        py::call_guard<py::gil_scoped_release>());
  // Return each payload as py::bytes; pybind11 utf-8-decodes std::string
  // by default, corrupting binary payloads (e.g. msgpack).
  m.def("drain_recorded_completions", []() {
    auto items = drain_recorded_completions();
    py::list out;
    for (auto& kv : items) {
      out.append(py::make_tuple(py::str(kv.first), py::bytes(kv.second)));
    }
    return out;
  });
}
