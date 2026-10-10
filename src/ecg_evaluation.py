"""Independent calibration, development thresholds, patient-cluster intervals."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


def fit_platt(logits, y):
    logits, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both outcomes")

    def objective(w):
        z = w[0] + w[1] * logits
        residual = expit(z) - y
        return np.mean(np.logaddexp(0, z) - y * z), np.array([residual.mean(), np.mean(residual * logits)])

    result = minimize(objective, [np.log(y.mean() / (1 - y.mean())), 0.], jac=True, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 1000})
    if not np.isfinite(result.x).all() or (not result.success and np.max(np.abs(result.jac)) > 1e-5):
        raise ValueError(f"calibration failed: {result.message}")
    return result.x


def working_threshold(logits, y, sensitivity=.95):
    positive = np.asarray(logits)[np.asarray(y) == 1]
    if not len(positive):
        raise ValueError("working-point selection requires positive development examples")
    # Lower quantile and >= retain tied positives and never relax target sensitivity.
    return float(np.quantile(positive, 1 - sensitivity, method="lower"))


def metrics(y, p, decision, prevalence):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    decision = np.asarray(decision, dtype=bool)
    tp, fn = int((decision & (y == 1)).sum()), int((~decision & (y == 1)).sum())
    fp, tn = int((decision & (y == 0)).sum()), int((~decision & (y == 0)).sum())
    both = len(np.unique(y)) == 2
    ece, curve = 0., []
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        mask = bins == b
        if mask.any():
            obs, pred = float(y[mask].mean()), float(p[mask].mean())
            ece += mask.mean() * abs(obs - pred)
            curve.append({"bin": b, "n": int(mask.sum()), "mean_probability": pred, "observed_fraction": obs})
    return {"records": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "auprc_average_precision": float(average_precision_score(y, p)) if both else None,
            "brier": float(np.mean((p - y) ** 2)),
            "constant_train_prevalence_brier": float(np.mean((prevalence - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
            "ece_10bin": float(ece), "calibration_curve": curve,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None, "npv": tn / (tn + fn) if tn + fn else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}}


def cluster_intervals(rows, y, p, decision, prevalence, replicates=1000, seed=20261010):
    patients = np.asarray([r["patient_id"] for r in rows])
    unique, inverse = np.unique(patients, return_inverse=True)
    indices = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc_average_precision", "brier", "log_loss", "sensitivity", "specificity", "ppv", "npv")
    samples = {name: [] for name in names}
    for _ in range(replicates):
        ids = np.concatenate([indices[i] for i in rng.integers(0, len(unique), len(unique))])
        result = metrics(y[ids], p[ids], decision[ids], prevalence)
        for name in names:
            if result[name] is not None:
                samples[name].append(result[name])
    return {"unit": "patient_cluster", "replicates": replicates, "seed": seed,
            "ci95": {k: [float(v) for v in np.quantile(values, [.025, .975])] if values else None
                     for k, values in samples.items()}}


def report_target(rows, y, raw_logits, probability, threshold, prevalence, replicates, seed):
    decision = raw_logits >= threshold
    report = metrics(y, probability, decision, prevalence)
    report["patients"] = len({r["patient_id"] for r in rows})
    report["raw_selection_threshold"] = threshold
    report["intervals"] = cluster_intervals(rows, y, probability, decision, prevalence, replicates, seed)
    if len(np.unique(y)) == 2 and np.std(probability) > 1e-8:
        intercept, slope = fit_platt(np.log(np.clip(probability, 1e-12, 1 - 1e-12) /
                                                  np.clip(1 - probability, 1e-12, 1)), y)
        report["evaluation_calibration_intercept"] = float(intercept)
        report["evaluation_calibration_slope"] = float(slope)
    subgroups = {}
    masks = {"gender_0": [r["gender"] == "0" for r in rows], "gender_1": [r["gender"] == "1" for r in rows],
             "age_lt65": [float(r["age"]) < 65 for r in rows], "age_ge65": [float(r["age"]) >= 65 for r in rows],
             "interval_le6h": [float(r["Time_Interval"]) <= 360 for r in rows],
             "interval_le24h": [float(r["Time_Interval"]) <= 1440 for r in rows],
             "consistent_ami_subtype": [r["ami_subtype_disagreement"] == "0" for r in rows],
             "qc_no_flags": [not r["quality_flags"] for r in rows]}
    for field in ("Prior_PCI", "Paced", "VF_VT"):
        for value in ("0", "1"):
            masks[f"{field}_{value}"] = [r[field] == value for r in rows]
    seen = set()
    single = []
    for row in rows:
        single.append(row["patient_id"] not in seen)
        seen.add(row["patient_id"])
    masks["one_record_per_patient_index_order"] = single
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            subgroups[name] = metrics(y[mask], probability[mask], decision[mask], prevalence)
            subgroups[name]["patients"] = len({r["patient_id"] for r, keep in zip(rows, mask) if keep})
    report["subgroups_descriptive_without_intervals"] = subgroups
    return report
