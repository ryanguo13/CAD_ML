"""Verify raw ECG inference agrees with the float16 cache and trained head."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from ecg_data import ROOT, load_partition, read_csv, read_waveform, save_json
from ecg_models import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/e1_ecgfm_frozen_omi_v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((args.run / "protocol.json").read_text())
    experiment = protocol["config"]
    cpc = experiment["model"].startswith("ecgcpc")
    if cpc:
        from foundation_ecgcpc import config, embeddings, load_encoder, preprocessing
    else:
        from foundation_ecgfm import config, embeddings, load_encoder, preprocessing
    cfg = config()
    cache = ROOT / experiment["cache_dir"]
    rows = load_partition(ROOT / experiment["data_dir"], "train")[:2]
    index = {r["record_id"]: int(r["cache_row"]) for r in read_csv(cache / "index.csv")}
    stored = np.load(cache / "embeddings.npy", mmap_mode="r")
    encoder = load_encoder(cfg)
    checks = []
    for row in rows:
        inputs = preprocessing(read_waveform(row["waveform_path"]))
        raw = embeddings(encoder, inputs[None] if cpc else inputs, cfg).astype(np.float16).astype(np.float32)
        cached = stored[index[row["record_id"]]][None].astype(np.float32)
        relative_l2 = float(np.linalg.norm(raw - cached) / max(np.linalg.norm(cached), 1e-12))
        # Batched CUDA AMP may differ slightly; reject differences above 0.2%.
        assert relative_l2 < .002, (row["record_id"], relative_l2)
        if experiment["model"] == "ecgcpc_mean":
            raw, cached = raw.mean(1), cached.mean(1)
        for seed in experiment["seeds"]:
            checkpoint = torch.load(args.run / f"seed_{seed}_best.pt", map_location="cpu", weights_only=True)
            head = make_model(experiment["model"], checkpoint["targets"]).to(cfg["device"]).eval()
            head.load_state_dict(checkpoint["state_dict"])
            c = json.loads((args.run / f"seed_{seed}_calibration.json").read_text())["OMI"]
            with torch.inference_mode(), torch.autocast(device_type=cfg["device"], enabled=experiment["amp"]):
                a = float(head(torch.from_numpy(raw).to(cfg["device"])).float()[0, 0])
                b = float(head(torch.from_numpy(cached).to(cfg["device"])).float()[0, 0])
            pa, pb = expit(c["intercept"] + c["slope"] * np.array([a, b]))
            assert abs(pa - pb) < .001, (row["record_id"], seed, pa, pb)
            checks.append({"record_id": row["record_id"], "seed": seed, "embedding_relative_l2": relative_l2,
                           "logit_absolute_difference": abs(a - b), "probability_absolute_difference": abs(pa - pb)})
    report = {"status": "passed", "samples_from_training_only": True, "checks": checks,
              "embedding_relative_l2_limit": .002, "probability_absolute_limit": .001}
    save_json(args.run / "raw_vs_cache_inference.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
