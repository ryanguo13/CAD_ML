"""Patient-paired bootstrap differences on identical records and seed budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_data import ROOT, save_json, sha256


def scores(y, probability):
    return np.array([roc_auc_score(y, probability), average_precision_score(y, probability),
                     np.mean((y - probability) ** 2)])


def compare(candidate, reference, out, replicates=1000):
    if out.exists():
        raise FileExistsError("paired comparison already exists")
    cp = json.loads((candidate / "protocol.json").read_text())
    rp = json.loads((reference / "protocol.json").read_text())
    if cp["manifest_sha256"] != rp["manifest_sha256"] or cp["config"]["seeds"] != rp["config"]["seeds"]:
        raise ValueError("comparison requires identical patients and seed budgets")
    result = {"candidate": str(candidate), "reference": str(reference),
              "difference": "candidate_minus_reference", "brier_lower_is_better": True,
              "bootstrap_unit": "paired_patient_cluster", "replicates": replicates, "seeds": {}}
    all_deltas, truth, patients = [], None, None
    for seed in cp["config"]["seeds"]:
        a = json.loads((candidate / f"seed_{seed}_evaluation_predictions.json").read_text())
        b = json.loads((reference / f"seed_{seed}_evaluation_predictions.json").read_text())
        a, b = [r for r in a if r["target"] == "OMI"], [r for r in b if r["target"] == "OMI"]
        if [(r["record_id"], r["patient_id"], r["label"]) for r in a] != [(r["record_id"], r["patient_id"], r["label"]) for r in b]:
            raise ValueError("paired predictions do not refer to identical records")
        y = np.asarray([r["label"] for r in a])
        pa, pb = np.asarray([r["probability"] for r in a]), np.asarray([r["probability"] for r in b])
        patients = np.asarray([r["patient_id"] for r in a])
        unique, inverse = np.unique(patients, return_inverse=True)
        groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
        rng = np.random.default_rng(20261010)
        deltas = []
        for _ in range(replicates):
            ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if len(np.unique(y[ids])) == 2:
                deltas.append(scores(y[ids], pa[ids]) - scores(y[ids], pb[ids]))
        all_deltas.append(np.asarray(deltas))
        result["seeds"][str(seed)] = {name: {"difference": float(value),
                            "ci95": [float(v) for v in np.quantile(np.asarray(deltas)[:, i], [.025, .975])]}
                            for i, (name, value) in enumerate(zip(("auroc", "auprc", "brier"), scores(y, pa) - scores(y, pb)))}
    mean_delta = np.mean(all_deltas, axis=0)
    result["mean_over_predeclared_seeds_ci95"] = {name: [float(v) for v in np.quantile(mean_delta[:, i], [.025, .975])]
                                                for i, name in enumerate(("auroc", "auprc", "brier"))}
    result["prediction_file_sha256"] = {str(directory / f"seed_{seed}_evaluation_predictions.json"):
                                        sha256(directory / f"seed_{seed}_evaluation_predictions.json")
                                        for directory in (candidate, reference) for seed in cp["config"]["seeds"]}
    save_json(out, result)
    print(json.dumps(result["mean_over_predeclared_seeds_ci95"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/e1_ecgfm_frozen_omi_v1")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/e0_resnet_omi_v1_run")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/e1_ecgfm_frozen_omi_v1/paired_vs_e0.json")
    args = parser.parse_args()
    compare(args.candidate, args.reference, args.out)
