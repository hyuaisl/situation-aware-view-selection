"""60-second driving scenarios split by the REAL situation ratio of the data (Sec. 4.3).

Ratio = situation shares over all validation clips (every 0.5 s). Each scenario = 120
decisions (60 s at 0.5 s); each situation's quota is filled with contiguous runs of that
situation cut from held-out TEST clips (never used to pick schedules), no moment used twice.
Segment order inside a scenario is shuffled; the model sees only the last 0.4 s at each
decision, so the order does not change any result.

  python scripts/06_make_scenarios.py
  -> results/scenarios.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import config  # noqa: E402

STEPS = 120


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeline", default=config.result("timeline_all.csv"))
    p.add_argument("--out", default=config.result("scenarios.csv"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    tl = pd.read_csv(a.timeline)
    share = tl.situation.value_counts(normalize=True)
    raw = share * STEPS
    quota = np.floor(raw).astype(int)
    for s in (raw - quota).sort_values(ascending=False).index[: STEPS - quota.sum()]:
        quota[s] += 1
    print("ratio (%):", (share * 100).round(1).to_dict(), "-> decisions per 60 s:", quota.to_dict())

    test = tl[tl.split == "test"].sort_values(["clip_id", "k"]).reset_index(drop=True)
    runs = []  # contiguous same-situation runs inside one clip
    for _, g in test.groupby("clip_id"):
        cur = []
        for _, r in g.iterrows():
            if cur and (r.situation != cur[-1].situation or r.k - cur[-1].k != 5):
                runs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            runs.append(cur)
    rng = np.random.default_rng(a.seed)
    pool = {s: [run for run in runs if run[0].situation == s] for s in quota.index}
    for s in pool:
        rng.shuffle(pool[s])

    rows, n = [], 0
    while True:
        segs, ok = [], True
        for s, q in quota.items():
            need = q
            while need > 0 and pool[s]:
                run = pool[s].pop()
                take, rest = run[:need], run[need:]
                if rest:
                    pool[s].append(rest)
                segs.append(take)
                need -= len(take)
            if need > 0:
                ok = False
                break
        if not ok:
            break
        t = 0.0
        for j, i in enumerate(rng.permutation(len(segs))):
            for r in segs[i]:
                rows.append(dict(scenario=n, segment=j, clip_id=r.clip_id, k=int(r.k), t_sec=t,
                                 situation=r.situation, past_yaw=r.past_yaw, v_now=r.v_now, future=r.future))
                t += 0.5
        n += 1

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"{n} scenarios x {STEPS} = {len(out)} decisions from {out.clip_id.nunique()} held-out clips")


if __name__ == "__main__":
    main()
