"""Small, dependency-light ECG baseline for the local angiography-linked dataset.

This is a research baseline only. It deliberately uses raw ECG waveforms and
patient-level deterministic splits; no diagnosis, PCI, or angiography metadata is
used as an input feature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TARGETS = ("OMI", "AMI", "ACS", "CTO", "UA", "STEMI", "NSTEMI")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def patient_bucket(patient_id: str) -> int:
    return int(hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def features(path: Path) -> np.ndarray:
    """Read this dataset's validated WFDB format-16 layout and 8 stats/lead.

    This intentionally supports only the observed format, failing on unfamiliar
    encodings instead of silently reading another WFDB layout incorrectly.
    """
    lines = path.with_suffix(".hea").read_text(encoding="ascii").splitlines()
    head = lines[0].split()
    if head[1:4] != ["12", "500", "5000"]:
        raise ValueError(f"unsupported WFDB shape: {path}")
    signals = [line.split() for line in lines[1:13]]
    if len(signals) != 12 or any(s[1] != "16" or s[2] != "1000.0(0)/mV" or s[4] != "0" for s in signals):
        raise ValueError(f"unsupported WFDB encoding/gain/baseline: {path}")
    if len({s[0] for s in signals}) != 1 or signals[0][0] != path.name:
        raise ValueError(f"unsupported WFDB signal filenames: {path}")
    lead_names = [s[-1] for s in signals]
    if set(lead_names) != set(LEADS):
        raise ValueError(f"unsupported lead names: {path}")
    x = np.fromfile(path, dtype="<i2")
    if x.size != 12 * 5000 or np.any(x == -32768):
        raise ValueError(f"missing samples or unexpected waveform size: {path}")
    x = x.reshape(5000, 12).T[[lead_names.index(s) for s in LEADS]].astype(np.float32) / 1000.0
    d = np.diff(x, axis=1)
    return np.concatenate(
        [x.mean(1), x.std(1), np.percentile(x, 10, axis=1),
         np.percentile(x, 90, axis=1), x.min(1), x.max(1),
         d.std(1), np.mean(np.abs(d), axis=1)]
    ).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 700, lr: float = 0.08):
    # Standardized features plus an intercept. L2 regularization keeps the
    # intentionally small baseline stable with imbalanced labels.
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    weights = np.where(y > 0, 0.5 / pos, 0.5 / neg).astype(np.float32)
    zz = np.c_[np.ones(len(z), dtype=np.float32), z]
    w = np.zeros(zz.shape[1], dtype=np.float64)
    for _ in range(steps):
        p = sigmoid(zz @ w)
        grad = (zz.T @ ((p - y) * weights)) + 1e-3 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return mu, sd, w


def fit_calibrator(logit: np.ndarray, y: np.ndarray, steps: int = 500):
    """Fit a 2-parameter logistic calibration map on validation data."""
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("calibration requires both outcomes")
    design = np.c_[np.ones(len(logit)), logit]
    w = np.array([np.log(y.mean() / (1 - y.mean())), 0.0])
    for _ in range(min(steps, 50)):
        p = sigmoid(design @ w)
        gradient = design.T @ (p - y)
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + np.eye(2) * 1e-7
        change = np.linalg.solve(hessian, gradient)
        w -= change
        if np.max(np.abs(change)) < 1e-8:
            break
    return float(w[0]), float(w[1])


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    # Mann-Whitney form; ties receive their average rank.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    s = 0
    while s < len(p):
        e = s + 1
        while e < len(p) and p[order[e]] == p[order[s]]:
            e += 1
        ranks[order[s:e]] = (s + e - 1) / 2.0 + 1
        s = e
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    out = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1 / bins
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)


def load_rows(csv_path: Path, wave_dir: Path):
    rows, xs, excluded = [], [], []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            path = wave_dir / row["ecg_row_record"]
            try:
                values = features(path)
            except ValueError as error:
                excluded.append(str(error))
                continue
            rows.append(row)
            xs.append(values)
            if len(rows) % 4000 == 0:
                print(f"Read {len(rows)} ECGs", flush=True)
    return rows, np.asarray(xs, dtype=np.float32), excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(r"D:\新建文件夹\CSV\train.csv"))
    ap.add_argument("--wave-dir", type=Path, default=Path(r"D:\新建文件夹\row_data"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\CAD_ML\artifacts\baseline_raw"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, x, excluded = load_rows(args.csv, args.wave_dir)
    (args.out / "excluded_records.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    buckets = np.array([patient_bucket(r["Patient_id"]) for r in rows])
    # 0-79 train, 80-89 calibration, 90-99 untouched test; all records for a
    # patient stay together. The boundaries are fixed for reproducibility.
    split = {"train": buckets < 80, "calibration": (buckets >= 80) & (buckets < 90), "test": buckets >= 90}
    patient_sets = {k: {r["Patient_id"] for r, selected in zip(rows, mask) if selected} for k, mask in split.items()}
    assert not (patient_sets["train"] & patient_sets["calibration"] or patient_sets["train"] & patient_sets["test"] or patient_sets["calibration"] & patient_sets["test"])
    result = {"rows": len(rows), "features": int(x.shape[1]), "input": "raw WFDB 12-lead 500 Hz 10 s, mV", "split_method": "SHA256(patient_id), buckets 0:80/80:90/90:100", "split_rows": {k: int(v.sum()) for k, v in split.items()}, "split_patients": {k: len(v) for k,v in patient_sets.items()}, "interpretation": "record-level internal exploratory performance; no external validation or patient-cluster confidence intervals", "targets": {}}
    result["excluded_records"] = len(excluded)
    model = {}
    for target in TARGETS:
        print(f"Training {target}", flush=True)
        y = np.asarray([int(r["AMI"] == "1" or r["UA"] == "1") if target == "ACS" else int(r[target]) for r in rows], dtype=np.float32)
        mu, sd, w = fit_logistic(x[split["train"]], y[split["train"]])
        def logits(mask):
            return np.c_[np.ones(mask.sum()), (x[mask] - mu) / sd] @ w
        cal_a, cal_b = fit_calibrator(logits(split["calibration"]), y[split["calibration"]])
        test_logit = logits(split["test"])
        p = sigmoid(cal_a + cal_b * test_logit)
        yt = y[split["test"]]
        result["targets"][target] = {
            "train_positive": int(y[split["train"]].sum()),
            "calibration_positive": int(y[split["calibration"]].sum()),
            "test_positive": int(yt.sum()),
            "test_prevalence": float(yt.mean()),
            "test_auroc": auc(yt, p),
            "test_brier": float(np.mean((p - yt) ** 2)),
            "constant_train_prevalence_brier": float(np.mean((y[split["train"]].mean() - yt) ** 2)),
            "test_ece_10bin": ece(yt, p),
            "calibration_intercept": cal_a,
            "calibration_slope": cal_b,
        }
        model[target] = {"mu": mu, "sd": sd, "w": w, "calibration": np.array([cal_a, cal_b])}
    np.savez(args.out / "baseline_model.npz", **{f"{t}_{k}": v for t, m in model.items() for k, v in m.items()})
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
