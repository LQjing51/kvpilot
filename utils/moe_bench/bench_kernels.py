"""Prefill time of one prompt at a time, as the prefill worker runs it (one
sequence, chunked prefill at 8192, cudagraphs for chunks up to 1024), for one
MoE kernel / expert-parallel choice; optionally a torch profile.

    python3 bench_kernels.py <model> <tp> <ep 0|1> <moe_backend|auto> [profile_dir]

Prints `RESULT ... tokens=N ms=M` per prompt length (median of 3 after a
warm-up) and, with a profile dir, the profiler's CUDA time table there.
"""
import os
import random
import statistics
import sys
import time


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model, tp, ep, backend = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
    prof_dir = sys.argv[5] if len(sys.argv) > 5 else None
    kw = {}
    if backend != "auto":
        kw["moe_backend"] = backend
    if prof_dir:
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": prof_dir,
                                 "torch_profiler_with_stack": False,
                                 "torch_profiler_dump_cuda_time_total": True}
    comp = {"cudagraph_capture_sizes": [64, 128, 256, 512, 1024]}
    # FI_AR_MB: the flashinfer fused allreduce size limit (Hopper defaults are
    # 64 MB at TP 2 but 2 MB at TP 4, so TP 4 always falls back to NCCL).
    if "FI_AR_MB" in os.environ:
        comp["pass_config"] = {"fi_allreduce_fusion_max_size_mb": float(os.environ["FI_AR_MB"])}
    llm = LLM(model=model, tensor_parallel_size=tp, enable_expert_parallel=ep,
              language_model_only=True, max_model_len=65536,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
              max_num_seqs=1, max_num_batched_tokens=8192, enable_prefix_caching=False,
              compilation_config=comp, **kw)
    sp = SamplingParams(max_tokens=1)
    random.seed(0)
    tag = (f"model={model.rstrip('/').split('/')[-1]} tp={tp} ep={int(ep)} backend={backend}"
           f" fi_ar_mb={os.environ.get('FI_AR_MB', 'default')}")
    prompt = lambda n: [TokensPrompt(prompt_token_ids=[random.randrange(1000, 100000) for _ in range(n)])]
    if prof_dir:
        for n in (1024, 8192):
            llm.generate(prompt(n), sp, use_tqdm=False)
        llm.start_profile()
        for n in (1024, 8192):
            llm.generate(prompt(n), sp, use_tqdm=False)
        llm.stop_profile()
        print(f"PROFILED {tag} -> {prof_dir}", flush=True)
        return
    for n in (256, 1024, 2048, 4096, 8192):
        ts = []
        for _ in range(4):
            ids = prompt(n)
            t = time.perf_counter()
            llm.generate(ids, sp, use_tqdm=False)
            ts.append((time.perf_counter() - t) * 1e3)
        print(f"RESULT {tag} tokens={n} ms={statistics.median(ts[1:]):.1f}", flush=True)


if __name__ == "__main__":
    main()
