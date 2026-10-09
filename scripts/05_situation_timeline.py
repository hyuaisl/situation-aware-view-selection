"""Situation of every validation clip every 0.5 s (no model run). Used for the real-data
situation ratio and as the pool of moments for the driving scenarios (Sec. 4.3).

  python scripts/05_situation_timeline.py
  -> results/timeline_all.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import common as C, config  # noqa: E402
from svs.situations import past_features, situation, split_of  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", default=config.PREP)
    p.add_argument("--stride", type=int, default=5, help="grid steps (0.1 s) between moments")
    p.add_argument("--out", default=config.result("timeline_all.csv"))
    a = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.prep, "*.pkl"))):
        pack = C.load_pack(path)
        cid = pack["clip_id"]
        for k in list(C.valid_t0_indices(pack))[:: a.stride]:
            py, v, fut = past_features(pack, k)
            rows.append(dict(clip_id=cid, split=split_of(cid), k=k, past_yaw=py, v_now=v, future=fut,
                             situation=situation(py, v)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "situation"]).size().unstack())
    print("share over all clips (%):", (df.situation.value_counts(normalize=True) * 100).round(1).to_dict())


if __name__ == "__main__":
    main()
