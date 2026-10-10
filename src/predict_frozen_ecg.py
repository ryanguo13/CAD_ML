"""Audited frozen ECG-FM/CPC encoder and calibrated head, from raw mV ECG."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from ecg_data import ROOT, quality_flags, read_waveform, sha256
from ecg_models import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/e1_ecgfm_frozen_omi_v1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((args.run / "protocol.json").read_text())
    trained_config = protocol["config"]
    if trained_config["model"] == "ecgfm_head":
        from foundation_ecgfm import config, embeddings, load_encoder, preprocessing
        source_root = ROOT / "external/fairseq-signals-master"
        source_hash_key = "fairseq_source_hashes"
        adapter = ROOT / "src/foundation_ecgfm.py"
    elif trained_config["model"] in ("ecgcpc_query", "ecgcpc_mean"):
        from foundation_ecgcpc import SOURCE, config, embeddings, load_encoder, preprocessing
        source_root, source_hash_key = SOURCE, "official_source_hashes"
        adapter = ROOT / "src/foundation_ecgcpc.py"
    else:
        raise ValueError("run is not a supported frozen encoder experiment")
    cache = ROOT / trained_config["cache_dir"]
    if sha256(cache / "manifest.json") != protocol["cache_manifest_sha256"]:
        raise ValueError("head and frozen cache provenance differ")
    cfg = config()
    cache_lock = json.loads((cache / "lock.json").read_text())
    if cfg != cache_lock["config"]:
        raise ValueError("inference preprocessing differs from training cache")
    if sha256(adapter) != cache_lock["adapter_sha256"]:
        raise ValueError("inference adapter differs from frozen cache extraction")
    for name, expected in cache_lock[source_hash_key].items():
        if sha256(source_root / name) != expected:
            raise ValueError("official encoder source differs from frozen cache extraction")
    checkpoint_path = args.run / f"seed_{args.seed}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["cache_manifest_sha256"] != protocol["cache_manifest_sha256"]:
        raise ValueError("prediction head provenance differs")
    calibration = json.loads((args.run / f"seed_{args.seed}_calibration.json").read_text())
    x = read_waveform(args.waveform)
    encoder = load_encoder(cfg)
    inputs = preprocessing(x)
    if trained_config["model"].startswith("ecgcpc"):
        inputs = inputs[None]
    vector = embeddings(encoder, inputs, cfg).astype(np.float16).astype(np.float32)
    if trained_config["model"] == "ecgcpc_mean":
        vector = vector.mean(1)
    # Match the stored float16 representation used to train the head.
    head = make_model(checkpoint["model"], checkpoint["targets"]).to(cfg["device"]).eval()
    head.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode(), torch.autocast(device_type=cfg["device"], enabled=trained_config["amp"]):
        logits = head(torch.from_numpy(vector).to(cfg["device"])).float().cpu().numpy()[0]
    output = {"model": checkpoint["model"], "seed": args.seed, "head_sha256": sha256(checkpoint_path),
              "foundation_sha256": cfg["checkpoint_sha256"], "quality_flags": quality_flags(x),
              "quality_status": "engineering_screen_only_clinical_QC_not_validated",
              "scope": protocol["evaluation_status"], "cad_probability": None, "probabilities": {}}
    for target, logit in zip(checkpoint["targets"], logits):
        c = calibration[target]
        output["probabilities"][target] = float(expit(c["intercept"] + c["slope"] * logit))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
