"""Runnable numerical and split-isolation checks; no clinical performance claim."""
import tempfile
from pathlib import Path

import numpy as np
from baseline_ecg import auc, features, fit_calibrator, patient_bucket, sigmoid


def main():
    assert auc(np.array([0, 1]), np.array([0.2, 0.9])) == 1.0
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert auc(np.array([1, 1]), np.array([0.2, 0.8])) is None
    assert patient_bucket("patient-A") == patient_bucket("patient-A")
    logits = np.repeat(np.array([-2., -1., 0., 1., 2.]), 100)
    truth = np.concatenate([np.r_[np.ones(n), np.zeros(100-n)] for n in [5, 10, 20, 40, 60]])
    a, b = fit_calibrator(logits, truth)
    calibrated = sigmoid(a + b * logits)
    assert abs(calibrated.mean() - truth.mean()) < 1e-6
    assert b > 0
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / "invalid.dat"
        p.with_suffix(".hea").write_text("invalid 1 500 5000\n", encoding="ascii")
        p.write_bytes(b"\0\0")
        try:
            features(p)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported data must fail closed")
    print("AUC ties, calibration intercept, deterministic grouping, and input rejection: passed")


if __name__ == "__main__":
    main()
