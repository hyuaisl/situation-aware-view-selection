"""Scenario results (Table 2): four inputs on the same decisions.

CIs: bootstrap (10,000) over scenarios for the overall numbers, over clips per situation
(neighbouring decisions are correlated, so single decisions are not resampled).

  python scripts/08_scenario_analyze.py
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

CONDS = ["full", "adaptive", "FLFR", "random"]
PAIRS = [("adaptive", "full"), ("FLFR", "full"), ("random", "full"), ("adaptive", "FLFR"), ("adaptive", "random")]
rng = np.random.default_rng(0)


def cluster_boot(w, col, unit, n=10000):
    g = w.groupby(unit)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=config.result("raw", "scenario_eval.jsonl"))
    p.add_argument("--scenarios", default=config.result("scenarios.csv"))
    a = p.parse_args()

    sc = pd.read_csv(a.scenarios)[["scenario", "t_sec", "clip_id", "k", "situation"]]
    df = pd.DataFrame([json.loads(l) for l in open(a.runs)]).drop(columns=["scenario"], errors="ignore")
    df = df.merge(sc, on=["clip_id", "k"]).drop_duplicates(["clip_id", "k", "cond"])

    for metric in ["min_ade", "fde"]:
        w = df.pivot_table(index=["scenario", "clip_id", "k", "situation"], columns="cond",
                           values=metric).dropna().reset_index()
        for x, y in PAIRS:
            w[f"{x}-{y}"] = w[x] - w[y]
        print(f"\n=== {metric}: {len(w)} decisions, {w.scenario.nunique()} scenarios, {w.clip_id.nunique()} clips ===")
        print("mean:", {c: round(float(w[c].mean()), 3) for c in CONDS})
        for x, y in PAIRS:
            print(f"  {x+'-'+y:<16} by scenario {fmt(*cluster_boot(w, f'{x}-{y}', 'scenario'))}")
        print("  per situation (bootstrap over clips):")
        for sit in SITUATIONS:
            g = w[w.situation == sit]
            means = " ".join(f"{c}={g[c].mean():.2f}" for c in CONDS)
            diffs = " | ".join(f"{x[0]}-{y}: {fmt(*cluster_boot(g, f'{x}-{y}', 'clip_id'))}"
                               for x, y in PAIRS if x == "adaptive")
            print(f"   {sit:<7} n={len(g):<5} {means} | {diffs}")
        per = w.groupby("scenario")[CONDS].mean()
        print("  scenarios where adaptive is lower than full / FLFR / random:",
              int((per.adaptive < per.full).sum()), int((per.adaptive < per.FLFR).sum()),
              int((per.adaptive < per.random).sum()), "of", len(per))

    lat = df.groupby("cond").ms_5080
    print("\ninference time per decision, ms: mean", lat.mean().round(1).to_dict(),
          "| median", lat.median().round(1).to_dict())
    ad = df[df.cond == "adaptive"]
    print("situation decision time, ms: median", round(ad.select_ms.median(), 3), "max", round(ad.select_ms.max(), 3))
    print("schedules used by adaptive:", ad.sched.value_counts().to_dict())


if __name__ == "__main__":
    main()
