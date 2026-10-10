"""Official ECG-FM loading, audited preprocessing, and resumable frozen cache."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_data import ROOT, SPLITS, read_csv, read_waveform, save_json, sha256, verify_manifest, write_csv


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config(path=None):
    return json.loads((path or ROOT / "configs/foundation_ecgfm.json").read_text(encoding="utf-8"))


def preprocessing(x):
    """Use official transform class, with the official standardize-then-cut order."""
    from ecg_transform.inp import ECGInput
    from ecg_transform.sample import ECGMetadata
    from ecg_transform.t.scale import Standardize
    metadata = ECGMetadata(sample_rate=500, num_samples=5000,
                           lead_names=list(config()["preprocessing"]["lead_order"]),
                           unit="mV", input_start=0, input_end=5000)
    standardized = Standardize()._transform(ECGInput(x, metadata)).signal
    windows = np.stack((standardized[:, :2500], standardized[:, 2500:]), axis=0).astype(np.float32)
    if windows.shape != (2, 12, 2500) or not np.isfinite(windows).all():
        raise ValueError("invalid ECG-FM input windows")
    return windows


def load_encoder(cfg):
    checkpoint = ROOT / cfg["checkpoint"]
    if sha256(checkpoint) != cfg["checkpoint_sha256"]:
        raise ValueError("ECG-FM checkpoint does not match the official HuggingFace LFS SHA256")
    for filename, expected in (("ecg-fm-source.zip", cfg["official_repository_commit"]),
                               ("fairseq-signals-source.zip", cfg["fairseq_signals_commit"])):
        with zipfile.ZipFile(ROOT / "artifacts" / filename) as archive:
            if archive.comment.decode() != expected:
                raise ValueError(f"official source revision mismatch: {filename}")
    sys.path.insert(0, str(ROOT / "external/fairseq-signals-master"))
    from fairseq_signals.models import build_model_from_checkpoint
    model = build_model_from_checkpoint(str(checkpoint)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.cfg.encoder_embed_dim != cfg["embedding_dim_per_window"]:
        raise ValueError("actual checkpoint embedding dimension differs from locked protocol")
    return model.to(cfg["device"])


@torch.inference_mode()
def embeddings(model, windows, cfg):
    source = torch.from_numpy(windows).to(cfg["device"])
    with torch.autocast(device_type=cfg["device"], enabled=cfg["amp"]):
        result = model.extract_features(source=source, padding_mask=None, mask=False)
    # Final Transformer output x, not pre-Transformer unmasked features.
    tokens = result["x"].float()
    mask = result["padding_mask"]
    if mask is not None:
        tokens = tokens.masked_fill(mask[:, :, None], 0)
    denominator = (tokens != 0).sum(dim=1)
    if (denominator == 0).any():
        raise ValueError("empty embedding readout")
    pooled = tokens.sum(dim=1) / denominator
    out = pooled.reshape(-1, 2 * cfg["embedding_dim_per_window"]).cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError("nonfinite frozen embeddings")
    return out


def extract(args):
    cfg = config(args.config)
    data, out = ROOT / cfg["data_dir"], ROOT / cfg["cache_dir"]
    verify_manifest(data)
    records = [r for r in read_csv(data / "records.csv") if r["split"] in SPLITS]
    if not torch.cuda.is_available() and cfg["device"] == "cuda":
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    model = load_encoder(cfg)
    dependency_versions = {p: importlib.metadata.version(p) for p in
                           ("torch", "numpy", "ecg-transform", "hydra-core", "omegaconf", "transformers")}
    lock = {"config": cfg, "manifest_sha256": sha256(data / "manifest.json"),
            "preprocessing_hash": fingerprint(cfg["preprocessing"]), "readout_version": cfg["readout_version"],
            "checkpoint_sha256": cfg["checkpoint_sha256"], "dependencies": dependency_versions,
            "adapter_sha256": sha256(Path(__file__)),
            "fairseq_source_hashes": {str(p.relative_to(ROOT / "external/fairseq-signals-master")): sha256(p)
                                      for p in (ROOT / "external/fairseq-signals-master/fairseq_signals").rglob("*.py")},
            "official_notebook_sha256": sha256(ROOT / "external/ECG-FM-main/notebooks/infer_quickstart.ipynb"),
            "record_count": len(records), "embedding_shape": [len(records), 1536]}
    if args.smoke:
        selected = records[:2]
        x = np.concatenate([preprocessing(read_waveform(r["waveform_path"])) for r in selected])
        output = embeddings(model, x, cfg)
        again = embeddings(model, x, cfg)
        if output.shape != (2, 1536) or not np.array_equal(output, again):
            raise AssertionError("frozen output shape or deterministic eval failed")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        embeddings(model, x, cfg)
        torch.cuda.synchronize()
        smoke = {"status": "passed", "parameters": sum(p.numel() for p in model.parameters()),
                 "encoder_layers": model.cfg.encoder_layers, "encoder_embed_dim": model.cfg.encoder_embed_dim,
                 "shape": list(output.shape), "seconds_two_records": time.perf_counter() - start,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "checkpoint_vs_yaml": "checkpoint_internal_configuration_is_authoritative"}
        print(json.dumps(smoke, indent=2), flush=True)
        save_json(ROOT / "artifacts/foundation/ecgfm/smoke.json", smoke)
        return
    if out.exists():
        if json.loads((out / "lock.json").read_text()) != lock:
            raise ValueError("cache provenance changed; choose a new version")
        if (out / "manifest.json").exists():
            verify_cache(out, data)
            print("Frozen cache already complete and verified", flush=True)
            return
        state = json.loads((out / "status.json").read_text())
        start_index = state["completed_records"]
        array = np.lib.format.open_memmap(out / "embeddings.npy", mode="r+")
    else:
        out.mkdir(parents=True)
        save_json(out / "lock.json", lock)
        write_csv(out / "index.csv", [{"cache_row": i, "record_id": r["record_id"], "split": r["split"],
                                      "waveform_sha256": r["waveform_sha256"], "header_sha256": r["header_sha256"]}
                                     for i, r in enumerate(records)])
        array = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float16, shape=(len(records), 1536))
        start_index = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for lo in range(start_index, len(records), cfg["batch_size_records"]):
        selected = records[lo:lo + cfg["batch_size_records"]]
        windows = []
        for row in selected:
            path = Path(row["waveform_path"])
            if sha256(path) != row["waveform_sha256"] or sha256(path.with_suffix(".hea")) != row["header_sha256"]:
                raise ValueError("waveform changed after data lock")
            windows.append(preprocessing(read_waveform(path)))
        array[lo:lo + len(selected)] = embeddings(model, np.concatenate(windows), cfg).astype(np.float16)
        array.flush()
        save_json(out / "status.json", {"status": "extracting", "completed_records": lo + len(selected)})
        if lo % 400 == 0:
            print(f"ECG-FM cached {lo + len(selected)}/{len(records)} ECGs", flush=True)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite float16 cache")
    save_json(out / "manifest.json", {"lock_sha256": sha256(out / "lock.json"),
                "file_sha256": {f: sha256(out / f) for f in ("index.csv", "embeddings.npy")},
                "record_count": len(records), "dtype": "float16", "shape": list(array.shape),
                "seconds_this_session": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "frozen_parameters": sum(p.numel() for p in model.parameters()),
                "hidden_records_used": 0})
    save_json(out / "status.json", {"status": "complete", "completed_records": len(records)})
    print("ECG-FM frozen cache complete", flush=True)


def verify_cache(cache, data):
    cache, data = Path(cache), Path(data)
    verify_manifest(data)
    manifest = json.loads((cache / "manifest.json").read_text())
    if sha256(cache / "lock.json") != manifest["lock_sha256"]:
        raise ValueError("embedding provenance modified")
    lock = json.loads((cache / "lock.json").read_text())
    if sha256(data / "manifest.json") != lock["manifest_sha256"]:
        raise ValueError("embedding cache and data lock differ")
    for name, expected in manifest["file_sha256"].items():
        if sha256(cache / name) != expected:
            raise ValueError(f"embedding cache modified: {name}")
    index = read_csv(cache / "index.csv")
    original = {r["record_id"]: r for r in read_csv(data / "records.csv") if r["split"] in SPLITS}
    if len(index) != len(original) or len({r["record_id"] for r in index}) != len(index):
        raise ValueError("missing or repeated cached records")
    for i, row in enumerate(index):
        if int(row["cache_row"]) != i or any(row[k] != original[row["record_id"]][k]
                                              for k in ("split", "waveform_sha256", "header_sha256")):
            raise ValueError("embedding/record identity mismatch")
    values = np.load(cache / "embeddings.npy", mmap_mode="r")
    if list(values.shape) != manifest["shape"] or values.dtype != np.float16 or not np.isfinite(values).all():
        raise ValueError("invalid frozen cache values")
    return manifest


class EmbeddingDataset(Dataset):
    def __init__(self, rows, targets, cache):
        self.rows, self.targets = rows, targets
        self.values = np.load(Path(cache) / "embeddings.npy", mmap_mode="r")
        self.index = {r["record_id"]: int(r["cache_row"]) for r in read_csv(Path(cache) / "index.csv")}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = self.values[self.index[row["record_id"]]].astype(np.float32)
        y = np.asarray([float(row[t]) if row[t] else np.nan for t in self.targets], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundation_ecgfm.json")
    parser.add_argument("--smoke", action="store_true")
    extract(parser.parse_args())
