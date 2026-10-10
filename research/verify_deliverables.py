"""Check deliverable consistency and record provenance without changing data."""
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np

root = Path(__file__).resolve().parents[1]
for path in (root / "src").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"))
plan = (root / "docs/ECG_CAD项目规划.md").read_text(encoding="utf-8")
assert "\ufffd" not in plan
for _, link in re.findall(r"\[([^]]+)\]\(([^)]+)\)", plan):
    if not link.startswith("http"):
        assert (root / "docs" / link).resolve().exists(), link
metrics = json.loads((root / "artifacts/baseline_raw/metrics.json").read_text())
assert sum(metrics["split_rows"].values()) == metrics["rows"] == 17955
assert metrics["excluded_records"] == 5
with np.load(root / "artifacts/baseline_raw/baseline_model.npz", allow_pickle=False) as model:
    assert len(model.files) == 28
    assert all(np.isfinite(model[k]).all() for k in model.files)
provenance = {
    "date": "2026-10-09",
    "numpy": np.__version__,
    "source": "https://doi.org/10.6084/m9.figshare.29925314",
    "csv_sha256": hashlib.sha256(Path(r"D:\新建文件夹\CSV\train.csv").read_bytes()).hexdigest(),
    "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "src").glob("*.py")},
    "scope": "exploratory internal baseline; no prospective or external clinical validation",
    "features": ["mean", "std", "p10", "p90", "min", "max", "diff_std", "diff_abs_mean"],
    "order": "feature-major, then I II III aVR aVL aVF V1 V2 V3 V4 V5 V6",
    "training": "700 full-batch steps; inverse-class weighting; L2=1e-3; learning rate=0.08",
    "calibration": "two-parameter logistic fit on independent calibration split",
    "unavailable_target": "CAD",
    "limitations": ["no convergence study", "no clinical fiducial features", "limited quality control", "no patient-cluster confidence intervals", "independent heads lack clinical hierarchy"],
}
(root / "artifacts/baseline_raw/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
print("Passed: Python syntax, document local links, model finite values, split totals; provenance saved.")
print(f"Plan: {len(plan)} characters")
