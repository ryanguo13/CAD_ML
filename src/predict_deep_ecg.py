"""Single-record inference from a trained checkpoint and independent calibrator."""
import argparse
import json
from pathlib import Path

from scipy.special import expit
import torch

from ecg_data import ROOT, quality_flags, read_waveform, sha256
from ecg_models import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("waveform", type=Path)
    ap.add_argument("--run", type=Path, default=ROOT / "artifacts/e0_resnet_omi_v1")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    checkpoint_path = args.run / f"seed_{args.seed}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    calibrators = json.loads((args.run / f"seed_{args.seed}_calibration.json").read_text(encoding="utf-8"))
    if checkpoint["architecture_sha256"] != sha256(ROOT / "configs/model_architecture_v1.json"):
        raise ValueError("architecture specification differs from trained checkpoint")
    x = read_waveform(args.waveform)
    model = make_model(checkpoint["model"], checkpoint["targets"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        logits = model(torch.from_numpy(x[None]).to(args.device)).float().cpu().numpy()[0]
    output = {"checkpoint_sha256": sha256(checkpoint_path), "units": "mV", "quality_flags": quality_flags(x),
              "quality_status": "engineering_screen_only_clinical_QC_not_validated",
              "scope": "research_probability_in_similar_angiography_cohort", "probabilities": {},
              "cad_probability": None, "cad_unavailable_reason": "no_complete_CAD_reference_labels"}
    for target, logit in zip(checkpoint["targets"], logits):
        cal = calibrators[target]
        output["probabilities"][target] = float(expit(cal["intercept"] + cal["slope"] * logit))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
