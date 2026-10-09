"""Paths used by every script. Override any of them with an environment variable.

SVS_WORK        working directory holding data/, results/, logs/        (default: current dir)
SVS_PREP        prepared validation clips (one .pkl per clip)            (default: $SVS_WORK/data/prep/val)
SVS_RESULTS     where result files are written                           (default: $SVS_WORK/results)
SVS_MODEL_CODE  model code: tools/eval_ade.py and alpamayo2/src          (default: $SVS_WORK/code)
SVS_CKPT        model checkpoint (.pt)                                   (default: $SVS_WORK/ckpt/final.pt)
SVS_BACKBONE    vision-language backbone config dir                      (default: $SVS_MODEL_CODE/cosmos2b)
SVS_COC_DATA    reasoning (CoC) data dir the model loader expects        (default: $SVS_MODEL_CODE/data/coc_34b_v1)
"""

import os

WORK = os.path.abspath(os.environ.get("SVS_WORK", "."))
PREP = os.environ.get("SVS_PREP", os.path.join(WORK, "data", "prep", "val"))
RESULTS = os.environ.get("SVS_RESULTS", os.path.join(WORK, "results"))
MODEL_CODE = os.environ.get("SVS_MODEL_CODE", os.path.join(WORK, "code"))
CKPT = os.environ.get("SVS_CKPT", os.path.join(WORK, "ckpt", "final.pt"))
BACKBONE = os.environ.get("SVS_BACKBONE", os.path.join(MODEL_CODE, "cosmos2b"))
COC_DATA = os.environ.get("SVS_COC_DATA", os.path.join(MODEL_CODE, "data", "coc_34b_v1"))


def result(*parts: str) -> str:
    """Path under the results directory (parent folders are created)."""
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
