"""Predict historical dataset labels, with research-only probability metadata."""
import argparse
import json
from pathlib import Path

import numpy as np
from baseline_ecg import TARGETS, features, sigmoid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path, help="WFDB .dat, matching .hea required")
    parser.add_argument("--model", type=Path, default=Path(r"D:\CAD_ML\artifacts\baseline_raw\baseline_model.npz"))
    args = parser.parse_args()
    values = features(args.waveform)
    probabilities = {}
    with np.load(args.model, allow_pickle=False) as model:
        for target in TARGETS:
            mu, sd, w = (model[f"{target}_{k}"] for k in ("mu", "sd", "w"))
            a, b = model[f"{target}_calibration"]
            logit = np.r_[1.0, (values - mu) / sd] @ w
            probabilities[target] = float(sigmoid(a + b * logit))
    print(json.dumps({
        "status": "research_only",
        "population": "single-center pre-angiography dataset; not externally validated",
        "targets": "historical angiography/discharge labels; not clinical diagnoses",
        "probabilities": probabilities,
        "cad_probability": None,
        "cad_unavailable_reason": "CAD-positive and verified CAD-negative reference labels are unavailable",
        "limitations": "Independent marginal outputs may violate hierarchy; do not add probabilities. Basic format checks are not a validated signal-quality assessment."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
