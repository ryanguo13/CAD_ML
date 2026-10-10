"""Verify standardize-before-cut, frozen-head scale isolation and cache identity."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from ecg_data import ROOT, save_json, sha256, write_csv
from ecg_models import FrozenEmbeddingHead
from foundation_ecgfm import preprocessing, verify_cache
from train_ecg import fit_embedding_scale


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 5000)).astype(np.float32)
    x[:, 2500:] += 3
    windows = preprocessing(x)
    joined = np.concatenate([windows[0], windows[1]], axis=1)
    assert np.allclose(joined.mean(1), 0, atol=1e-6)
    assert np.allclose(joined.std(1), 1, atol=1e-6)
    assert abs(windows[0].mean()) > .5  # catches accidental per-window z-score
    model = FrozenEmbeddingHead()
    train = torch.from_numpy(rng.normal(size=(3, 1536)).astype(np.float32))
    fit_embedding_scale(model, [(train, None)], "cpu")
    assert torch.allclose(model.feature_mean, train.mean(0), atol=1e-6)
    model(train).sum().backward()
    assert model.feature_mean.grad is None
    # A modified index must be rejected before any head can use the cache.
    with tempfile.TemporaryDirectory() as folder:
        cache = Path(folder)
        save_json(cache / "lock.json", {})
        write_csv(cache / "index.csv", [{"record_id": "a", "split": "train"}])
        save_json(cache / "manifest.json", {"lock_sha256": sha256(cache / "lock.json"),
                                            "file_sha256": {"index.csv": sha256(cache / "index.csv")}})
        (cache / "lock.json").write_text('{"changed": true}')
        try:
            verify_cache(cache, ROOT / "artifacts/data_v1")
        except ValueError as e:
            assert "provenance modified" in str(e)
        else:
            raise AssertionError("modified provenance must be rejected")
    print("Passed: native normalization order, train-only scaling, head gradients, provenance rejection")


if __name__ == "__main__":
    main()
