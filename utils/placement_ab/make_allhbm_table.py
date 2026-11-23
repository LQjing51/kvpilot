"""A signal table that makes every request HBM safe: each turn's next_tier
(and own_tier) becomes HBM, keeping the measured next_time, so penalties still
order demotions. The last turn of a session stays without a next turn.

    python3 make_allhbm_table.py signal_swe_reqprio.json signal_swe_reqprio_allhbm.json

Under exclusive placement every block then wants to stay in HBM: on a trace
whose live KV exceeds the HBM pool, HBM stays full of held blocks and every
admission has to make room (run_ab.sh dataset swereqhbm).
"""
import json
import sys
from collections import Counter


def main(src: str, out: str) -> None:
    table = json.load(open(src))
    for row in table.values():
        if row["next_tier"] is not None:
            row["next_tier"] = "HBM"
        if row["own_tier"] is not None:
            row["own_tier"] = "HBM"
    json.dump(table, open(out, "w"))
    print(out, Counter(r["next_tier"] for r in table.values()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
