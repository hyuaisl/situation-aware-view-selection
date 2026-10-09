"""Inference time vs number of input images (Fig. 1), from the recorded runs.

4 images  = the 81 four-image schedules      (results/raw/combo_sweep.jsonl)
12 images = full input, 3 cameras x 4 slots  (results/raw/combo_sweep.jsonl)
16 / 24   = full4 / full6 with extra cameras (results/raw/latency_runs.jsonl, see 03_combo_sweep.py)

  python scripts/09_latency_by_images.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[config.result("raw", "combo_sweep.jsonl"),
                                                 config.result("raw", "latency_runs.jsonl")])
    a = p.parse_args()

    ms: dict[int, list[float]] = {}
    for path in a.runs:
        for line in open(path):
            r = json.loads(line)
            ms.setdefault(r["n_img"], []).append(r["ms_5080"])
    n = np.array(sorted(ms))
    med = np.array([np.median(ms[i]) for i in n])
    for i, m in zip(n, med):
        print(f"{i:>3} images: median {m:7.1f} ms  ({len(ms[i])} runs)")
    slope, icpt = np.polyfit(n, med, 1)
    print(f"linear fit: {icpt:.0f} ms + {slope:.1f} ms per image")


if __name__ == "__main__":
    main()
