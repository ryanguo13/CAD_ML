"""Small executable checks for isolation, units, missing labels, and backprop."""
import tempfile
from pathlib import Path

import numpy as np
import torch
import wfdb

from ecg_data import grouped_split, read_waveform
from ecg_evaluation import fit_platt, working_threshold
from ecg_models import FusionModel, LocalModel, ResNet, architecture, masked_bce


def main():
    torch.set_num_threads(4)
    records = [dict(patient_id=f"p{i // 2}", waveform_sha256=f"h{i}", OMI=i % 2, AMI=1) for i in range(80)]
    records[2]["waveform_sha256"] = records[0]["waveform_sha256"]
    split = grouped_split(records)
    assert split["p0"] == split["p1"]
    assert split == grouped_split(records)
    with tempfile.TemporaryDirectory() as folder:
        leads = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]
        raw = np.tile(np.arange(12)[None, :] * 100, (5000, 1)) + (np.arange(5000)[:, None] % 5)
        wfdb.wrsamp("sample", fs=500, units=["mV"] * 12, sig_name=leads, d_signal=raw,
                    fmt=["16"] * 12, adc_gain=[2000.] * 12, baseline=[100] * 12, write_dir=folder)
        decoded = read_waveform(Path(folder) / "sample.dat")
        assert decoded.shape == (12, 5000)
        assert np.isclose(decoded[0, 0], .5) and np.isclose(decoded[-1, 0], -.05)
        bad = Path(folder) / "sample.dat"
        corrupt = np.fromfile(bad, dtype="<i2")
        corrupt[0] = -32768
        corrupt.tofile(bad)
        try:
            read_waveform(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("missing samples must be rejected")
    z = torch.tensor([[0., 2.], [1., -1.]], requires_grad=True)
    y = torch.tensor([[1., float("nan")], [0., float("nan")]])
    masked_bce(z, y, [1., .5]).backward()
    assert torch.all(z.grad[:, 1] == 0)
    torch.manual_seed(17)
    x = torch.randn(1, 12, 5000) * .1
    config = architecture()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = {}
    for name, model in (("resnet", ResNet()), ("local", LocalModel(config)), ("fusion", FusionModel(config))):
        model = model.to(device)
        output = model(x.to(device), torch.randn(1, 1536, device=device)) if name == "fusion" else model(x.to(device))
        assert output.shape == (1, 1) and torch.isfinite(output).all()
        masked_bce(output, torch.ones_like(output)).backward()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count < 1_000_000
        checks[name] = count
    summary = LocalModel(config).local.amplitude_summary(x)
    scaled = LocalModel(config).local.amplitude_summary(2 * x)
    assert torch.allclose(scaled, 2 * summary, atol=1e-6)
    assert working_threshold(np.array([.1, .2, .3]), np.ones(3)) == .1
    labels = np.tile([0., 1.], 100)
    a, b = fit_platt(np.tile([-1., 1.], 100), labels)
    assert np.isfinite([a, b]).all() and b > 0
    print({"checks": "passed", "device": device, "trainable_parameters": checks}, flush=True)


if __name__ == "__main__":
    main()
