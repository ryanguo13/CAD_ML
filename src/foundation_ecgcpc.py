"""Official CPC encoder/S4, with an exact PyTorch Cauchy backend for Windows."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import resampy
import torch
from torch import nn

from ecg_data import ROOT, SPLITS, read_csv, read_waveform, save_json, sha256, verify_manifest, write_csv

SOURCE = ROOT / "external/ecg-fm-benchmarking-main/code"


def config(path=None):
    return json.loads((path or ROOT / "configs/foundation_ecgcpc.json").read_text(encoding="utf-8"))


def official_imports():
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from clinical_ts.ts.s4_modules import s42
    # Only the Cauchy implementation changes; all layers and weights stay official.
    s42.has_cauchy_extension = False
    s42.cauchy_conj = cauchy_torch


def cauchy_torch(v, z, w):
    """Sum conjugate poles exactly, without a learned or approximate replacement."""
    z, w, v = z.unsqueeze(-2), w.unsqueeze(-1), v.unsqueeze(-1)
    return (v / (z - w) + v.conj() / (z - w.conj())).sum(dim=-2)


def preprocessing(x):
    if x.shape != (12, 5000) or not np.isfinite(x).all():
        raise ValueError("CPC requires a finite canonical 12-lead 10-second mV signal")
    y = resampy.resample(x, 500, 240, axis=1).astype(np.float32)
    if y.shape != (12, 2400) or not np.isfinite(y).all():
        raise ValueError("invalid native CPC input")
    return y


class CPCEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        official_imports()
        from clinical_ts.template_modules import ShapeConfig, StaticStats
        from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
        from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig
        shape = ShapeConfig(channels=12, length=2400, sequence_last=True)
        self.encoder = RNNEncoder(RNNEncoderConfig(**cfg["encoder"]), shape, StaticStats())
        self.predictor = S4Predictor(S4PredictorConfig(**cfg["predictor"]), self.encoder.get_output_shape())

    def forward(self, x):
        return self.predictor(**self.encoder(seq=x))["seq"]


def load_encoder(cfg):
    path = ROOT / cfg["checkpoint"]
    if sha256(path) != cfg["checkpoint_sha256"]:
        raise ValueError("CPC checkpoint SHA256 mismatch")
    with zipfile.ZipFile(ROOT / "artifacts/ecg-cpc-source.zip") as archive:
        if archive.comment.decode() != cfg["official_repository_commit"]:
            raise ValueError("CPC official source commit mismatch")
    model = CPCEncoder(cfg)
    # Trusted, hash-pinned official Lightning checkpoint contains OmegaConf classes.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("ts_encoder."): v for k, v in checkpoint["state_dict"].items()
             if k.startswith(("ts_encoder.encoder.", "ts_encoder.predictor."))}
    # Official buffers use expanded zero-stride views; assign avoids copy_ into
    # overlapping memory in newer PyTorch, while still requiring every key/shape.
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model.to(cfg["device"])


@torch.inference_mode()
def embeddings(model, x, cfg):
    # Native float32 avoids mixed-precision complex/FFT changes to the S4 encoder.
    out = model(torch.from_numpy(x).to(cfg["device"])).float().cpu().numpy()
    if out.shape != (len(x), 1200, 512) or not np.isfinite(out).all():
        raise ValueError("invalid official final S4 tokens")
    return out


def verify_cache(cache, data):
    cache, data = Path(cache), Path(data)
    verify_manifest(data)
    manifest = json.loads((cache / "manifest.json").read_text())
    if sha256(cache / "lock.json") != manifest["lock_sha256"]:
        raise ValueError("CPC cache provenance modified")
    lock = json.loads((cache / "lock.json").read_text())
    if sha256(data / "manifest.json") != lock["manifest_sha256"]:
        raise ValueError("CPC cache and data lock differ")
    for name, expected in manifest["file_sha256"].items():
        if sha256(cache / name) != expected:
            raise ValueError(f"CPC cache modified: {name}")
    index = read_csv(cache / "index.csv")
    original = {r["record_id"]: r for r in read_csv(data / "records.csv") if r["split"] in SPLITS}
    if len(index) != len(original) or len({r["record_id"] for r in index}) != len(index):
        raise ValueError("missing or repeated CPC records")
    for i, r in enumerate(index):
        if int(r["cache_row"]) != i or r["record_id"] not in original or any(
                r[k] != original[r["record_id"]][k] for k in ("split", "waveform_sha256", "header_sha256")):
            raise ValueError("CPC tokens/record identity mismatch")
    values = np.load(cache / "embeddings.npy", mmap_mode="r")
    if list(values.shape) != [len(index), 1200, 512] or values.dtype != np.float16:
        raise ValueError("invalid CPC cache shape/dtype")
    means = np.load(cache / "mean_embeddings.npy", mmap_mode="r")
    if means.shape != (len(index), 512) or means.dtype != np.float32 or not np.isfinite(means).all():
        raise ValueError("invalid CPC mean-pooling cache")
    # Chunked check keeps the 22 GB token cache within the host memory budget.
    for lo in range(0, len(index), 64):
        if not np.isfinite(values[lo:lo + 64]).all():
            raise ValueError("nonfinite CPC token cache")
    return manifest


def extract(args):
    cfg = config(args.config)
    data, out = ROOT / cfg["data_dir"], ROOT / cfg["cache_dir"]
    verify_manifest(data)
    records = [r for r in read_csv(data / "records.csv") if r["split"] in SPLITS]
    torch.set_num_threads(4)
    model = load_encoder(cfg)
    lock = {"config": cfg, "manifest_sha256": sha256(data / "manifest.json"),
            "adapter_sha256": sha256(Path(__file__)),
            "official_source_hashes": {str(p.relative_to(SOURCE)): sha256(p) for p in SOURCE.rglob("*.py")},
            "dependencies": {p: importlib.metadata.version(p) for p in
                             ("torch", "numpy", "resampy", "lightning", "pytorch-lightning", "einops", "opt-einsum", "omegaconf")}}
    if out.exists():
        if json.loads((out / "lock.json").read_text()) != lock:
            raise ValueError("CPC cache provenance changed; choose a new version")
        if (out / "manifest.json").exists():
            verify_cache(out, data)
            print("CPC cache complete and verified", flush=True)
            return
        start_index = json.loads((out / "status.json").read_text())["completed_records"]
        values = np.lib.format.open_memmap(out / "embeddings.npy", mode="r+")
        means = np.lib.format.open_memmap(out / "mean_embeddings.npy", mode="r+")
    else:
        out.mkdir(parents=True)
        save_json(out / "lock.json", lock)
        write_csv(out / "index.csv", [{"cache_row": i, "record_id": r["record_id"], "split": r["split"],
                    "waveform_sha256": r["waveform_sha256"], "header_sha256": r["header_sha256"]}
                    for i, r in enumerate(records)])
        values = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float16,
                                          shape=(len(records), 1200, 512))
        means = np.lib.format.open_memmap(out / "mean_embeddings.npy", mode="w+", dtype=np.float32,
                                         shape=(len(records), 512))
        start_index = 0
        save_json(out / "status.json", {"status": "extracting", "completed_records": 0})
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for lo in range(start_index, len(records), cfg["batch_size_records"]):
        selected = records[lo:lo + cfg["batch_size_records"]]
        inputs = []
        for row in selected:
            path = Path(row["waveform_path"])
            if sha256(path) != row["waveform_sha256"] or sha256(path.with_suffix(".hea")) != row["header_sha256"]:
                raise ValueError("raw ECG changed after data lock")
            inputs.append(preprocessing(read_waveform(path)))
        block = embeddings(model, np.stack(inputs), cfg).astype(np.float16)
        if not np.isfinite(block).all():
            raise ValueError("float16 CPC token cache overflow")
        values[lo:lo + len(selected)] = block
        means[lo:lo + len(selected)] = block.astype(np.float32).mean(1)
        values.flush()
        means.flush()
        save_json(out / "status.json", {"status": "extracting", "completed_records": lo + len(selected)})
        if lo % 400 == 0:
            print(f"CPC cached {lo + len(selected)}/{len(records)} ECGs", flush=True)
    save_json(out / "manifest.json", {"lock_sha256": sha256(out / "lock.json"),
                "file_sha256": {f: sha256(out / f) for f in ("index.csv", "embeddings.npy", "mean_embeddings.npy")},
                "record_count": len(records), "dtype": "float16", "shape": list(values.shape),
                "seconds_this_session": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "frozen_parameters": sum(p.numel() for p in model.parameters()), "hidden_records_used": 0})
    save_json(out / "status.json", {"status": "complete", "completed_records": len(records)})
    print("CPC final-token cache complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundation_ecgcpc.json")
    extract(parser.parse_args())
