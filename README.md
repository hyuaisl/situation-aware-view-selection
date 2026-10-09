# Situation-Aware Selection of Time–Camera View Combinations for Efficient VLA Driving Models

Code and results for the KSC 2026 paper
**주행 상황별 시간–카메라 관측 조합 선택을 통한 VLA 주행 모델의 추론 효율 개선**.

A vision–language–action (VLA) driving model usually receives every camera at every recent
frame. Here the model gets **one camera per frame slot**: for the last 4 slots
(t−0.3, t−0.2, t−0.1, t s), each slot shows the front, left or right camera, so the input
shrinks from 12 to 4 images. The camera for each slot is chosen per **driving situation**
(normal, turning left, turning right, slow), which is decided from past ego motion only.
The model weights are not changed.

## Main results

Driving scenarios built from held-out clips with the real situation ratio of the data
(22 scenarios × 60 s, a decision every 0.5 s, 2,640 decisions). RTX 5080, batch size 1.

| Input | Images | minADE₆ (m) | vs. full input (m), 95% CI | Inference time (ms) |
|---|---|---|---|---|
| Full input (3 cameras × 4 slots) | 12 | 3.82 | — | 717 |
| Random camera per slot | 4 | 3.95 | +0.13 [−0.03, +0.27] | 434 |
| Fixed front-left-front-right | 4 | 3.89 | +0.07 [−0.12, +0.24] | 431 |
| **Situation-wise selection** | 4 | **3.40** | **−0.42 [−0.59, −0.27]** | 433 |

Schedules chosen on the selection clips (slot order t−0.3 → t; F front, L left, R right):

| Situation | Rule (past ego motion) | Schedule |
|---|---|---|
| normal | otherwise | F F L F |
| turning left | heading changed > 5° to the left in the past 1.5 s | F F F F |
| turning right | heading changed > 5° to the right in the past 1.5 s | L F R L |
| slow | speed < 10 km/h | L R R L |

Full printouts: [`results/logs/`](results/logs).

## What is in this repository

```
svs/                      shared code
  config.py               paths (environment variables, see below)
  common.py               model loading, sample cutting, slot-labelled prompt, minADE
  schedules.py            schedule names <-> camera/slot views, random schedule
  situations.py           situation rule (10 km/h, 5° over 1.5 s), selection/held-out split
scripts/                  run in this order
results/                  outputs used in the paper (small files and raw per-run results)
```

| Paper | Script | Output |
|---|---|---|
| data | `00_download_pai.sh`, `01_prepare_clips.py` | `data/prep/val/*.pkl` (not included) |
| Sec. 4.1 samples | `02_pick_situation_samples.py` | `results/situ_samples.csv` |
| Sec. 4.2 (81 schedules) | `03_combo_sweep.py` (GPU) | `results/raw/combo_sweep.jsonl` |
| Sec. 4.2, Table 1 | `04_select_schedules.py` | `results/selection/`, `results/logs/04_select_schedules.txt` |
| Sec. 4.3 situation ratio | `05_situation_timeline.py` | `results/timeline_all.csv` |
| Sec. 4.3 scenarios | `06_make_scenarios.py` | `results/scenarios.csv` |
| Sec. 4.3 runs | `07_scenario_eval.py` (GPU) | `results/raw/scenario_eval.jsonl` |
| Table 2 | `08_scenario_analyze.py` | `results/logs/08_scenario_analyze.txt` |
| Fig. 1 | `03_combo_sweep.py --scheds full4 full6`, `09_latency_by_images.py` | `results/raw/latency_runs.jsonl`, `results/logs/09_latency_by_images.txt` |

The analysis scripts (`04`, `08`, `09`) run on the included results without a GPU.

## Not included

* **Model code and checkpoint.** The 2B Alpamayo-style model (`alpamayo2_super`, the loader
  `tools/eval_ade.py`, the backbone config and the stage-2 checkpoint) belong to the lab's
  training code and are not part of this repository. `svs/common.py` calls
  `eval_ade.build` and `eval_ade.expert_traj` from that code.
* **Data.** PhysicalAI-Autonomous-Vehicles (NVIDIA, Hugging Face), validation chunks 186–188
  (300 clips). The download script comes from an `alpamayo-recipes` checkout.

## Setup

```bash
pip install -r requirements.txt
export SVS_WORK=/path/to/workdir            # data/, results/ go here
export SVS_MODEL_CODE=/path/to/model/code   # contains tools/ and alpamayo2/src
export SVS_CKPT=/path/to/final.pt
export SVS_RECIPES=/path/to/alpamayo-recipes
```

Other paths (`SVS_PREP`, `SVS_RESULTS`, `SVS_BACKBONE`, `SVS_COC_DATA`) default to locations under
those; see [`svs/config.py`](svs/config.py).

```bash
bash scripts/00_download_pai.sh 186-188
python scripts/01_prepare_clips.py --chunks 186 187 188 --split val
python scripts/02_pick_situation_samples.py
python scripts/03_combo_sweep.py            # ~4.5 h on an RTX 5080
python scripts/04_select_schedules.py
python scripts/05_situation_timeline.py
python scripts/06_make_scenarios.py
python scripts/07_scenario_eval.py          # ~1.5 h
python scripts/08_scenario_analyze.py
```

## Notes

* **Selection vs. held-out clips.** Clips are split by a hash of the clip id
  (`svs/situations.py`). Schedules are chosen on the selection clips only; Table 2 uses only
  held-out clips. The thresholds (10 km/h, 5°) were fixed before any schedule was compared.
* **Frame labels.** The model's prompt names each image with its camera and frame number.
  The library numbers frames per camera with `enumerate`, which is wrong when a camera appears
  only at some slots, so `svs/common.py` labels each image with its real slot. For the full
  input the prompt is checked to be identical to the library's.
* **Same seed per moment.** All inputs at one moment use the same random seed, so differences
  come from the input only. Rerunning the GPU scripts reproduced the recorded minADE values.
* **Inference time** is one forward pass including reasoning-text generation and 6 trajectory
  candidates, batch size 1, measured per decision.
* **Confidence intervals.** Paired bootstrap (10,000). In Table 2, decisions are resampled by
  scenario (overall) or by clip (per situation), because neighbouring decisions are correlated.
