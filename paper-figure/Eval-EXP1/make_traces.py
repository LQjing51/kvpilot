"""Eval-EXP1 traces and their signal tables, one trace for all three models.

    python3 make_traces.py swe  pool=20,k=176,mid=1,seed=0,order=h24f25 --n 3000
    python3 make_traces.py swe  pool=10,k=160,mid=1,seed=1,order=rand --n 3000 --name swe_rand
    python3 make_traces.py tracelab pool=...,order=... --n ... [--name tl_rand]

A trace is search_trace.py's candidate (datasets/block-priority-trace,
chosen with search_trace_models.py) over the dataset's sessions:
gen_trace_capacity's SWE cache, or TraceLab's (tracelab_all.jsonl, sessions
of two turns or more whose prompts fit the services, built here). Writes
exp1_<dataset>.jsonl and per model the oracle signal table run_ab.sh reads
(make_model_table.py: the next turn's suffix, tier and times from that
model's time model):

    signal_exp1_<name>.json        Qwen3-30B-A3B, TP 2, block 512
    signal_exp1_<name>_q36f8.json  Qwen3.6-35B-A3B-FP8, TP 2, block 1536
    signal_exp1_<name>_q38.json    Qwen3.8-Flash-Next, TP 4, block 1024
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "datasets/block-priority-trace"))
import search_trace as st  # noqa: E402
import search_trace_models as stm  # noqa: E402

gtc, gt = st.gtc, st.gt
FIG3 = ROOT / "paper-figure/Figure3-request-safe"
CACHES = {"swe": Path("/tmp/gen_trace_capacity.pkl"), "tracelab": Path("/tmp/gen_trace_tracelab.pkl")}
SOURCES = {"swe": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_full.jsonl",
           "tracelab": ROOT / "datasets/agent-traces/tracelab/tracelab_all.jsonl"}
# Prompt + forced output a session may reach (max_model_len 262144 on all three).
MAX_TOKENS = 200_000
# (Tables of the first round's traces used Qwen3.6 BF16 at TP 4: "_b1024",
# fit_times2_qwen3.6-tp4_k160.json, block 1024.)
MODELS = (("", FIG3 / "fit_times2_swe.json", 512, False),
          ("_q36f8", FIG3 / "fit_times2_qwen3.6-fp8_k160.json", 1536, True),
          ("_q38", FIG3 / "fit_times2_qwen3.8-ep_k160.json", 1024, True))


def cache(ds):
    path = CACHES[ds]
    if not path.exists():
        sess = gt.load(SOURCES[ds])
        if ds == "tracelab":
            sess = {s: rows for s, rows in sess.items() if len(rows) >= 2
                    and max(r["input_length"] + r["output_length"] for r in rows) <= MAX_TOKENS}
        pickle.dump((sess, gtc.turn_plans(sess)), open(path, "wb"))
    return pickle.load(open(path, "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(CACHES))
    ap.add_argument("spec", help="pool=..,k=..,mid=..,seed=..,order=<search_trace_models order>")
    ap.add_argument("--n", type=int, required=True, help="requests to keep")
    ap.add_argument("--name", help="writes exp1_<name>.jsonl (default: the dataset)")
    a = ap.parse_args()
    name = a.name or a.dataset

    sess, plans = cache(a.dataset)
    fp = {s: gtc.foot(plans[s]) for s in sess if len(sess[s]) >= 2}
    order = sorted(fp, key=lambda s: -fp[s][0])
    kv = dict(x.split("=") for x in a.spec.split(","))
    pool = int(kv["pool"])
    st.G.update(sess=sess, plans=plans, file_order=[s for s in sess if len(sess[s]) >= 2],
                pools={pool: sorted(order[: max(1, len(order) * pool // 100)])} if pool else {})
    trace = st.candidate(pool, int(kv["k"]), int(kv["mid"]), int(kv["seed"]), a.n,
                         **stm.parse_order(kv["order"]))
    out = HERE / f"exp1_{name}.jsonl"
    gt.write(out, trace, sess)
    print(f"wrote {out}: {len(trace)} requests, {len({s for s, _ in trace})} sessions ({a.spec})")
    for suffix, coeffs, block, drops in MODELS:
        table = HERE / f"signal_exp1_{name}{suffix}.json"
        cmd = [sys.executable, str(ROOT / "utils/placement_ab/make_model_table.py"), str(out),
               str(coeffs), str(table), "--block", str(block)] + (["--drops-last"] if drops else [])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
