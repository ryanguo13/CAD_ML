"""Constant, demographics, and existing waveform-statistics controls on one lock."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_ecg import features
from ecg_data import ROOT, load_partition, save_json, sha256
from ecg_evaluation import fit_platt, report_target, working_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/experiment_e0.json")
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = ROOT / config["data_dir"]
    out = ROOT / config["out_dir"] / "controls"
    if out.exists():
        raise FileExistsError("controls already recorded; do not overwrite evaluation")
    out.mkdir(parents=True)
    rows = {s: load_partition(data, s) for s in ("train", "selection", "calibration", "evaluation")}
    x = {}
    for split, selected in rows.items():
        print(f"Extracting existing 96-dimension waveform statistics: {split}", flush=True)
        waveform = []
        for r in selected:
            path = Path(r["waveform_path"])
            if sha256(path) != r["waveform_sha256"] or sha256(path.with_suffix(".hea")) != r["header_sha256"]:
                raise ValueError("waveform changed after lock")
            waveform.append(features(path))
        x[split] = {"waveform_statistics": np.asarray(waveform),
                    "demographics": np.asarray([[float(r["age"]), float(r["gender"])] for r in selected])}
    reports = {"evaluation_status": config["evaluation_status"], "manifest_sha256": sha256(data / "manifest.json"),
               "input_exclusions": ["diagnoses", "PCI", "angiography", "Time_Interval"], "models": {}}
    for target in config["targets"]:
        y = {s: np.asarray([float(r[target]) for r in selected]) for s, selected in rows.items()}
        prevalence = float(y["train"].mean())
        constant = np.full(len(y["evaluation"]), prevalence)
        reports["models"].setdefault("constant", {})[target] = report_target(
            rows["evaluation"], y["evaluation"], np.zeros(len(constant)), constant, 0., prevalence,
            config["bootstrap_replicates"], config["bootstrap_seed"])
        for name in ("demographics", "waveform_statistics"):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=17))
            model.fit(x["train"][name], y["train"])
            z = {s: model.decision_function(x[s][name]) for s in ("selection", "calibration", "evaluation")}
            a, b = fit_platt(z["calibration"], y["calibration"])
            threshold = working_threshold(z["selection"], y["selection"], config["working_point_sensitivity"])
            reports["models"].setdefault(name, {})[target] = report_target(
                rows["evaluation"], y["evaluation"], z["evaluation"], expit(a + b * z["evaluation"]), threshold,
                prevalence, config["bootstrap_replicates"], config["bootstrap_seed"])
            scaler, estimator = model.steps[0][1], model.steps[1][1]
            np.savez(out / f"{name}_{target}.npz", mean=scaler.mean_, scale=scaler.scale_,
                     coefficient=estimator.coef_, intercept=estimator.intercept_, calibration=np.array([a, b]),
                     raw_threshold=threshold)
    save_json(out / "metrics.json", reports)
    print("Constant, demographics, and waveform-statistics controls evaluated on the same patients", flush=True)


if __name__ == "__main__":
    main()
