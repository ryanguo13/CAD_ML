"""Export ROC, precision-recall and calibration curves from saved predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from ecg_data import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-local", action="store_true")
    parser.add_argument("--include-cpc", action="store_true")
    args = parser.parse_args()
    runs = [("E0 ResNet", ROOT / "artifacts/e0_resnet_omi_v1_run"),
            ("E1 ECG-FM frozen", ROOT / "artifacts/e1_ecgfm_frozen_omi_v1")]
    if args.include_local:
        runs.append(("E3 Local", ROOT / "artifacts/e3_local_omi_v1"))
    if args.include_cpc:
        runs.extend([("E2 CPC query", ROOT / "artifacts/e2_ecgcpc_query_omi_v1"),
                     ("E2 CPC mean control", ROOT / "artifacts/e2_ecgcpc_mean_omi_v1")])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    colors = ("#687787", "#0065ad", "#d07800", "#21875d", "#a55d99")
    for (name, directory), color in zip(runs, colors):
        for i, seed in enumerate((17, 43, 101)):
            rows = [r for r in json.loads((directory / f"seed_{seed}_evaluation_predictions.json").read_text())
                    if r["target"] == "OMI"]
            y = np.asarray([r["label"] for r in rows])
            probability = np.asarray([r["probability"] for r in rows])
            label = name if i == 0 else None
            fpr, tpr, _ = roc_curve(y, probability)
            precision, recall, _ = precision_recall_curve(y, probability)
            observed, predicted = calibration_curve(y, probability, n_bins=8, strategy="quantile")
            axes[0].plot(fpr, tpr, color=color, alpha=.65, lw=1.4, label=label)
            axes[1].plot(recall, precision, color=color, alpha=.65, lw=1.4, label=label)
            axes[2].plot(predicted, observed, color=color, alpha=.65, lw=1.4, marker=".", label=label)
    axes[0].plot([0, 1], [0, 1], color="#aaa", ls="--", lw=1)
    axes[1].axhline(y.mean(), color="#aaa", ls="--", lw=1, label=f"Prevalence {y.mean():.3f}")
    axes[2].plot([0, 1], [0, 1], color="#aaa", ls="--", lw=1)
    for ax, title, xlabel, ylabel in zip(axes, ("ROC", "Precision-recall", "Calibration (quantile bins)"),
                                        ("False positive rate", "Recall", "Mean calibrated probability"),
                                        ("True positive rate", "Precision", "Observed positive fraction")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.15)
        ax.legend(loc="lower right" if ax is axes[0] else "best", frameon=False, fontsize=8)
    figure.suptitle("OMI: fixed internal angiography evaluation; each line is one predeclared seed", fontsize=11)
    out = ROOT / "artifacts/stage_b_summary"
    out.mkdir(exist_ok=True)
    suffix = "e0_e1_e3" if args.include_local else "e0_e1"
    if args.include_cpc:
        suffix += "_e2"
    for extension in ("png", "svg"):
        figure.savefig(out / f"omi_{suffix}.{extension}", dpi=180)
    print(f"Saved curves: {out / ('omi_' + suffix + '.png')}")


if __name__ == "__main__":
    main()
