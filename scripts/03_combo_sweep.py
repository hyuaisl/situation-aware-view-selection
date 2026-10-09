"""Run all 81 schedules (3 cameras ^ 4 slots) plus the full input on the situation samples.

Same sample, same seed for every schedule, so differences come only from the input. Rows are
appended to a jsonl; finished (sample, schedule) pairs are skipped, so a run can be resumed.

  python scripts/03_combo_sweep.py                                  # Sec. 4.2 (81 + full)
  python scripts/03_combo_sweep.py --scheds full4 full6 \\
      --out results/raw/latency_runs.jsonl                          # Fig. 1 (needs extra cameras)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import common as C, config  # noqa: E402
from svs import schedules as S  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=config.result("situ_samples.csv"))
    p.add_argument("--prep", default=config.PREP)
    p.add_argument("--out", default=config.result("raw", "combo_sweep.jsonl"))
    p.add_argument("--scheds", nargs="+", default=["full"] + S.ALL_81)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-traj", type=int, default=6)
    a = p.parse_args()

    dev = "cuda:0"
    torch.cuda.set_device(0)
    model, cfg, margs = C.build_model(dev)
    margs.samples = a.n_traj
    build = C.InputBuilder(cfg, model.tokenizer)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["clip_id"], r["k"], r["sched"]))
    samples = pd.read_csv(a.samples)
    total = len(samples) * len(a.scheds)
    print(f"{len(samples)} samples, {total} runs, {len(done)} already done", flush=True)

    checked, t_start, n_new = False, time.time(), 0
    with open(a.out, "a") as fout:
        pack, pack_id = None, None
        for i, s in samples.iterrows():
            if s.clip_id != pack_id:
                pack, pack_id = C.load_pack(os.path.join(a.prep, f"{s.clip_id}.pkl")), s.clip_id
            k = int(s.k)
            for name in a.scheds:
                if (s.clip_id, k, name) in done:
                    continue
                sample = C.make_sample(pack, k, S.views(name))
                inp, text = build(sample)
                if not checked and name == "full":  # patched prompt == library prompt for full input
                    if text != build.stock_text(sample):
                        sys.exit("prompt check FAILED: slot-labelled prompt differs from library prompt")
                    checked = True
                inp = C.to_device(inp, dev)
                torch.manual_seed(a.seed + k)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = C.predict(model, cfg, inp, margs, dev)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                ade, fde = C.min_ade(preds, sample["gt_xyz"])
                fout.write(json.dumps({
                    "clip_id": s.clip_id, "k": k, "situation": s.situation, "future": s.future,
                    "split": s.split, "sched": name, "min_ade": ade, "fde": fde, "ms_5080": ms,
                    "n_img": sum(len(f) for _, f in sample["views"])}) + "\n")
                n_new += 1
            fout.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t_start
                left = (total - len(done) - n_new) * el / max(n_new, 1)
                print(f"[{i+1}/{len(samples)} samples] {n_new} runs in {el/60:.1f} min, "
                      f"~{left/3600:.1f} h left", flush=True)
    print("sweep done", flush=True)


if __name__ == "__main__":
    main()
