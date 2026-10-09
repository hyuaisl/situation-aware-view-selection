"""Run the driving scenarios with four inputs (Sec. 4.3, Table 2).

At every decision (0.5 s apart) the model gets the last 4 slots (0.4 s):
  adaptive : situation from past ego motion -> the schedule chosen for it (04_select_schedules)
  full     : 3 cameras x 4 slots (12 images)
  FLFR     : fixed front, left, front, right (4 images)
  random   : one random camera per slot, reproducible per moment (4 images)
Same moment, same seed for all inputs. Finished (moment, input) pairs are skipped.

  python scripts/07_scenario_eval.py
  -> results/raw/scenario_eval.jsonl
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
from svs.situations import past_features, situation  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", default=config.PREP)
    p.add_argument("--scenarios", default=config.result("scenarios.csv"))
    p.add_argument("--choice", default=config.result("selection", "adaptive_choice.json"))
    p.add_argument("--out", default=config.result("raw", "scenario_eval.jsonl"))
    p.add_argument("--conds", nargs="+", default=["adaptive", "full", "FLFR", "random"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-traj", type=int, default=6)
    a = p.parse_args()

    choice = json.load(open(a.choice))
    print("schedule per situation:", choice, flush=True)
    samples = pd.read_csv(a.scenarios)
    print(f"{len(samples)} decisions in {samples.scenario.nunique()} scenarios", flush=True)

    dev = "cuda:0"
    torch.cuda.set_device(0)
    model, cfg, margs = C.build_model(dev)
    margs.samples = a.n_traj
    build = C.InputBuilder(cfg, model.tokenizer)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["clip_id"], r["k"], r["cond"]))

    t_start = time.time()
    with open(a.out, "a") as fout:
        pack, pack_id = None, None
        for i, s in samples.iterrows():
            if s.clip_id != pack_id:
                pack, pack_id = C.load_pack(os.path.join(a.prep, f"{s.clip_id}.pkl")), s.clip_id
            k = int(s.k)
            for cond in a.conds:
                if (s.clip_id, k, cond) in done:
                    continue
                t_sel = time.perf_counter()
                if cond == "adaptive":  # online: past ego motion -> situation -> chosen schedule
                    py, v, _ = past_features(pack, k)
                    sit = situation(py, v)
                    name = choice[sit]
                elif cond == "random":
                    sit, name = None, S.random_schedule(s.clip_id, k)
                else:
                    sit, name = None, cond
                sel_ms = (time.perf_counter() - t_sel) * 1000
                sample = C.make_sample(pack, k, S.views(name))
                inp, _ = build(sample)
                inp = C.to_device(inp, dev)
                torch.manual_seed(a.seed + k)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = C.predict(model, cfg, inp, margs, dev)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                ade, fde = C.min_ade(preds, sample["gt_xyz"])
                fout.write(json.dumps({
                    "scenario": int(s.scenario), "clip_id": s.clip_id, "k": k, "cond": cond,
                    "sched": name, "online_situation": sit, "select_ms": sel_ms,
                    "min_ade": ade, "fde": fde, "ms_5080": ms,
                    "n_img": sum(len(f) for _, f in sample["views"])}) + "\n")
            fout.flush()
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(samples)}] {(time.time()-t_start)/60:.1f} min", flush=True)
    print("scenario eval done", flush=True)


if __name__ == "__main__":
    main()
