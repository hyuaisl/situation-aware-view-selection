"""Pick one schedule per situation (Table 1) and check it on held-out clips (Sec. 4.2).

For every situation, the schedule with the lowest mean minADE6 difference to the full input
on the SELECTION (dev) clips is chosen among all 81. The chosen schedules are then evaluated
as they are on the held-out TEST clips (never used for the choice), with paired bootstrap CIs.

  python scripts/04_select_schedules.py
  -> results/selection/{adaptive_choice.json, schedule_by_situation.csv, test_eval.csv}
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import config  # noqa: E402
from svs.situations import SITUATIONS  # noqa: E402


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default=config.result("raw", "combo_sweep.jsonl"))
    p.add_argument("--out", default=os.path.join(config.RESULTS, "selection"))
    p.add_argument("--metric", default="min_ade", choices=["min_ade", "fde"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.DataFrame([json.loads(l) for l in open(a.sweep)])
    key = ["clip_id", "k"]
    full = df[df.sched == "full"].set_index(key)[a.metric].rename("full")
    s = df[df.sched != "full"].join(full, on=key)
    n_per = s.groupby(key).sched.nunique()
    s = s.set_index(key).loc[n_per[n_per == 81].index].reset_index()  # samples with all 81 runs
    s["delta"] = s[a.metric] - s["full"]
    meta = s.drop_duplicates(key)[key + ["situation", "split"]]
    print("samples:", meta.groupby(["situation", "split"]).size().unstack().to_dict())

    # every schedule's mean difference to the full input, per situation (all 450 samples)
    tab = s.pivot_table(index="sched", columns="situation", values="delta", aggfunc="mean")[SITUATIONS]
    tab["all"] = s.groupby("sched").delta.mean()
    tab.sort_values("all").round(3).to_csv(f"{a.out}/schedule_by_situation.csv")
    print("best schedules over all samples:", tab["all"].sort_values().head(3).round(3).to_dict())

    dev, test = s[s.split == "dev"], s[s.split == "test"]
    choice = {}
    for sit in SITUATIONS:
        t = dev[dev.situation == sit].groupby("sched").delta.mean().sort_values()
        choice[sit] = t.index[0]
        print(f"{sit:<7} -> {t.index[0]}  (selection {t.iloc[0]:+.3f}; next "
              + ", ".join(f"{i} {v:+.3f}" for i, v in t.iloc[1:4].items()) + ")")
    json.dump(choice, open(f"{a.out}/adaptive_choice.json", "w"), indent=2)

    rows = []
    ev = pd.concat([test[(test.situation == sit) & (test.sched == sc)] for sit, sc in choice.items()])
    for name, g in [("ALL", ev)] + [(sit, ev[ev.situation == sit]) for sit in SITUATIONS]:
        m, lo, hi = boot_ci(g.delta)
        rows.append(dict(group=name, n=len(g), full=g.full.mean(), diff=m, lo=lo, hi=hi))
        print(f"held-out {name:<7} n={len(g):<4} situation-wise - full: {fmt(m, lo, hi)}")
    pd.DataFrame(rows).to_csv(f"{a.out}/test_eval.csv", index=False)

    ms = df.groupby("sched").ms_5080.median()
    four = ms.drop("full")
    print(f"inference time (median): full {ms['full']:.0f} ms, 4-image schedules "
          f"{four.min():.0f}-{four.max():.0f} ms")


if __name__ == "__main__":
    main()
