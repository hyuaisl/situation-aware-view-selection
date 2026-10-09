"""Pick evaluation moments per driving situation (Sec. 4.1): 450 samples, split by clip.

normal 150, left 100, right 100, slow 100; at most 2 moments per clip and situation, at least
3 s apart. Situations come from past ego motion only (svs/situations.py). Clips are split into
selection ("dev") and held-out ("test") by a hash of the clip id, so no clip is on both sides.

  python scripts/02_pick_situation_samples.py
  -> results/situ_samples.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import common as C, config  # noqa: E402
from svs.situations import past_features, situation, split_of  # noqa: E402

TARGET = {"normal": 150, "left": 100, "right": 100, "slow": 100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", default=config.PREP)
    p.add_argument("--stride", type=int, default=5, help="candidate moment every N grid steps (0.1 s)")
    p.add_argument("--min-gap", type=int, default=30, help="min grid steps between picks in one clip")
    p.add_argument("--per-clip", type=int, default=2, help="max picks per clip per situation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=config.result("situ_samples.csv"))
    a = p.parse_args()

    cand = []
    for path in sorted(glob.glob(os.path.join(a.prep, "*.pkl"))):
        pack = C.load_pack(path)
        cid = pack["clip_id"]
        for k in list(C.valid_t0_indices(pack))[:: a.stride]:
            py, v, fut = past_features(pack, k)
            cand.append(dict(clip_id=cid, k=k, split=split_of(cid), past_yaw=py, v_now=v, future=fut,
                             situation=situation(py, v)))
    cand = pd.DataFrame(cand)
    print("candidates per situation:", cand.situation.value_counts().to_dict())

    picks = []
    for sit, n in TARGET.items():
        pool = cand[cand.situation == sit].sample(frac=1.0, random_state=a.seed)
        chosen, per_clip = [], {}
        for _, r in pool.iterrows():
            ks = per_clip.setdefault(r.clip_id, [])
            if len(ks) >= a.per_clip or any(abs(r.k - x) < a.min_gap for x in ks):
                continue
            ks.append(r.k)
            chosen.append(r)
            if len(chosen) >= n:
                break
        picks += chosen
        print(f"{sit:<7} picked {len(chosen)} / target {n}")
    df = pd.DataFrame(picks).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["situation", "split"]).size().unstack().to_string())
    print(f"saved {len(df)} -> {a.out}")


if __name__ == "__main__":
    main()
