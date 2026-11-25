"""A tuned Triton MoE table keyed for expert parallelism as it runs.

benchmark_moe.py --tune --enable-expert-parallel routes every token's top-k
among the local E/ep experts, so at batch M each local expert gets ep times the
tokens it gets in the engine (routing among all E, the others' masked). This
writes, for each M of the table, the config tuned at M/ep (the nearest key),
under the same file name in <out dir>.

    python3 shift_tuned.py <tuned json> <ep> <out dir>
"""
import json
import sys
from pathlib import Path


def main():
    src, ep, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    raw = json.loads(src.read_text())
    # Besides the batch sizes the table carries metadata ("triton_version").
    table = {int(k): v for k, v in raw.items() if k.isdigit()}
    keys = sorted(table)
    shifted = {k: v for k, v in raw.items() if not k.isdigit()}
    for m in keys:
        near = min(keys, key=lambda k: abs(k - m / ep))
        shifted[str(m)] = table[near]
        print(f"M={m}: config tuned at {near}")
    out.mkdir(parents=True, exist_ok=True)
    (out / src.name).write_text(json.dumps(shifted, indent=4) + "\n")


if __name__ == "__main__":
    main()
