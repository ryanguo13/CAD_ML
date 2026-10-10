"""Check exact Cauchy algebra and official pretrained S4 FFT/recurrence agreement."""
import copy
import time

import numpy as np
import torch

from ecg_data import ROOT, load_partition, read_waveform, save_json
from ecg_models import make_model, masked_bce
from foundation_ecgcpc import cauchy_torch, config, embeddings, load_encoder, preprocessing


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    # Independent algebraic expression used by the official PyKeOps kernel.
    v = torch.randn(2, 1, 3, 4, dtype=torch.cdouble)
    w = -torch.rand(3, 4).double() + 1j * torch.rand(3, 4).double()
    z = 1j * torch.linspace(-10, 10, 17).double()
    vv, ww, zz = v.unsqueeze(-1), w.unsqueeze(-1), z.unsqueeze(-2)
    expected = (2 * (zz * vv.real - (vv * ww.conj()).real) /
                ((zz - ww) * (zz - ww.conj()))).sum(-2)
    actual = cauchy_torch(v, z, w)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    cfg = config()
    head = make_model("ecgcpc_query", ["OMI"])
    head_loss = masked_bce(head(torch.randn(2, 1200, 512)), torch.tensor([[0.], [1.]]))
    head_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    head_parameters = sum(p.numel() for p in head.parameters())
    assert head_parameters == 787456 and head_parameters < 1_000_000
    model = load_encoder(cfg)
    errors = []
    with torch.inference_mode():
        for original in model.predictor.predictor.s4_layers:
            # Float64 kernel check avoids judging equivalence by roundoff alone.
            kernel = copy.deepcopy(original.kernel.kernel).cpu().double()
            fft, _ = kernel(L=1200)
            kernel.setup_step(mode="dense")
            state = kernel.default_state(1)
            impulse = torch.ones(1, 512, dtype=torch.float64)
            stepped = []
            for t in range(1200):
                y, state = kernel.step(impulse if t == 0 else torch.zeros_like(impulse), state)
                stepped.append(y[0].real)
            recurrent = torch.stack(stepped, -1)
            error = float((fft - recurrent).abs().max())
            relative_l2 = float(torch.linalg.vector_norm(fft - recurrent) / torch.linalg.vector_norm(recurrent))
            # Checkpoint FFT nodes are stored in float32, even for this float64
            # arithmetic check; their roundoff produces tiny absolute residuals.
            torch.testing.assert_close(fft, recurrent, rtol=1e-4, atol=1e-6)
            assert relative_l2 < 1e-4
            errors.append({"max_absolute": error, "relative_l2": relative_l2})
        rows = load_partition(ROOT / "artifacts/data_v1", "train")[:2]
        x = np.stack([preprocessing(read_waveform(r["waveform_path"])) for r in rows])
        tokens = embeddings(model, x, cfg)
        again = embeddings(model, x, cfg)
        assert np.array_equal(tokens, again)
        assert not any(p.requires_grad for p in model.parameters())
        # The official resampler is linear: physical mV scaling survives.
        np.testing.assert_allclose(preprocessing(2 * read_waveform(rows[0]["waveform_path"])),
                                   2 * x[0], rtol=1e-5, atol=1e-6)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        embeddings(model, np.repeat(x[:1], cfg["batch_size_records"], axis=0), cfg)
        torch.cuda.synchronize()
    report = {"status": "passed", "cauchy_algebra_max_abs_error": float((actual - expected).abs().max()),
              "four_pretrained_kernel_fft_vs_dense_recurrence_max_abs_errors": errors,
              "kernel_check_dtype": "float64", "kernel_check_steps": 1200,
              "parameters": sum(p.numel() for p in model.parameters()), "shape": list(tokens.shape),
              "official_query_head_parameters": head_parameters, "query_head_gradients": "passed",
              "seconds_eight_records": time.perf_counter() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "backend": cfg["cauchy_backend"], "strict_load": True,
              "configuration_note": cfg["configuration_authority"]}
    save_json(ROOT / "artifacts/foundation/ecgcpc/smoke.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
