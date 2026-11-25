"""GPU kernel time of a bench_kernels.py profile by category, per profiled
prefill (the `execute_context_1(N)` annotation of each step).

    python3 parse_prof.py <profile dir>   # reads rank 0's *.pt.trace.json.gz
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CATS = [  # first match wins
    ("MoE routed", r"fused_moe|moe_align|topkGating|topk_softmax|moe::|cutlass_moe|moe_gemm|expandInput|"
                   r"finalizeMoe|tensorrt_llm|trtllm::moe|grouped|deep_gemm|moe_sum|computeStridesTma|"
                   r"blockExpertPrefixSum|fusedBuildExpertMaps|globalExpertPrefixSum|mergeExpertPrefixSum"),
    ("Dense FP8 GEMM", r"cutlass_3x_gemm|nvjet|sm90_xmma|gemm|scaled_mm"),
    ("Linear attention", r"FlatKernel|gated_delta|chunk_|causal_conv1d|fused_post_conv|fla_|recurrent|gdn"),
    ("All-reduce", r"allreduce|all_reduce|nccl|two_shot|one_shot"),
    ("Attention", r"flash|fmha|attention|indexer|sparse"),
    ("FP8 quant", r"quant"),
]


def cat(name):
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "Other"


def main():
    d = Path(sys.argv[1])
    f = sorted(d.glob("*rank0*.pt.trace.json.gz"))[0]
    ev = json.load(gzip.open(f))["traceEvents"]
    steps = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
                   if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context"))
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    per = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        for t0, t1, name in steps:
            if t0 <= k["ts"] < t1:
                tokens = int(re.search(r"execute_context_\d+\((\d+)\)", name)[1])
                per[tokens][cat(k["name"])] += k["dur"] / 1e3
                break
    names = [c for c, _ in CATS] + ["Other"]
    for tokens in sorted(per):
        tot = sum(per[tokens].values())
        print(f"tokens={tokens}: GPU kernels {tot:.1f} ms | "
              + " | ".join(f"{c} {per[tokens][c]:.1f} ({per[tokens][c] / tot * 100:.0f}%)" for c in names))
    top = defaultdict(float)
    for k in kernels:
        top[k["name"][:90]] += k["dur"] / 1e3
    print("top kernels (all steps):")
    for n, v in sorted(top.items(), key=lambda x: -x[1])[:12]:
        print(f"  {v:7.1f} ms  [{cat(n)}] {n}")


if __name__ == "__main__":
    main()
