"""Reproducible E0/local training; evaluation is a separate explicit command."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader, Dataset

from ecg_data import ROOT, load_partition, read_waveform, save_json, sha256, verify_manifest
from ecg_evaluation import fit_platt, report_target, working_threshold
from ecg_models import LocalModel, make_model, masked_bce
from scipy.special import expit


class WaveDataset(Dataset):
    def __init__(self, rows, targets):
        self.rows, self.targets = rows, targets
        self.checked = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["waveform_path"])
        if index not in self.checked:
            if sha256(path) != row["waveform_sha256"] or sha256(path.with_suffix(".hea")) != row["header_sha256"]:
                raise ValueError(f"raw waveform changed after data lock: {row['record_id']}")
            self.checked.add(index)
        x = read_waveform(path)
        y = np.asarray([float(row[t]) if row[t] != "" else np.nan for t in self.targets], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def loader(rows, config, train=False):
    generator = torch.Generator().manual_seed(config.get("active_seed", 17))
    return DataLoader(WaveDataset(rows, config["targets"]), batch_size=config["batch_size"], shuffle=train,
                      num_workers=config["num_workers"], pin_memory=config["device"] == "cuda", generator=generator)


@torch.inference_mode()
def predict(model, batches, device, amp):
    model.eval()
    logits, truth = [], []
    for x, y in batches:
        with torch.autocast(device_type=device, enabled=amp):
            output = model(x.to(device, non_blocking=True))
        logits.append(output.float().cpu().numpy())
        truth.append(y.numpy())
    return np.concatenate(logits), np.concatenate(truth)


def fit_summary(model, batches, device):
    total = torch.zeros(48, dtype=torch.float64, device=device)
    squares = total.clone()
    count = 0
    with torch.inference_mode():
        for x, _ in batches:
            s = model.local.amplitude_summary(x.to(device)).double()
            total += s.sum(0)
            squares += s.square().sum(0)
            count += len(s)
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
        model.local.summary_mean.copy_(mean.float())
        model.local.summary_std.copy_(std.float().clamp_min(1e-6))


def train(config, config_path, out, data):
    if out.exists():
        raise FileExistsError("choose a new out_dir; previous experiments are not overwritten")
    manifest = verify_manifest(data)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by protocol but unavailable; do not silently change experiment hardware")
    out.mkdir(parents=True)
    save_json(out / "protocol.json", {"config": config, "config_sha256": sha256(config_path),
                                     "manifest_sha256": sha256(data / "manifest.json"),
                                     "architecture_sha256": sha256(ROOT / "configs/model_architecture_v1.json"),
                                     "code_sha256": {p.name: sha256(p) for p in (ROOT / "src").glob("*.py")},
                                     "dependencies": {"torch": torch.__version__, "numpy": np.__version__},
                                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                     "evaluation_status": manifest["evaluation_status"],
                                     "deterministic_algorithms": True})
    training = load_partition(data, "train")
    selection = load_partition(data, "selection")
    progress = {"status": "training", "completed_seeds": []}
    save_json(out / "status.json", progress)
    try:
        for seed in config["seeds"]:
            seed_all(seed)
            config_seed = {**config, "active_seed": seed}
            train_loader, select_loader = loader(training, config_seed, True), loader(selection, config_seed)
            model = make_model(config["model"], config["targets"]).to(config["device"])
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if parameters > 1_000_000:
                raise ValueError("trainable parameter budget exceeded")
            if isinstance(model, LocalModel):
                print("Fitting amplitude summary statistics on train patients only", flush=True)
                fit_summary(model, train_loader, config["device"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            amp = config["amp"] and config["device"] == "cuda"
            scaler = torch.amp.GradScaler(config["device"], enabled=amp)
            weights = [config["task_weights"][t] for t in config["targets"]]
            best_ap = -1.
            history = []
            if config["device"] == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, config["epochs"] + 1):
                start = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running = 0.
                for step, (x, y) in enumerate(train_loader):
                    x, y = x.to(config["device"], non_blocking=True), y.to(config["device"], non_blocking=True)
                    with torch.autocast(device_type=config["device"], enabled=amp):
                        loss = masked_bce(model(x), y, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    group_start = (step // config["accumulation_steps"]) * config["accumulation_steps"]
                    group_size = min(config["accumulation_steps"], len(train_loader) - group_start)
                    scaler.scale(loss / group_size).backward()
                    if (step + 1) % config["accumulation_steps"] == 0 or step + 1 == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    running += float(loss.detach()) * len(x)
                    if (step + 1) % 100 == 0:
                        print(f"seed={seed} epoch={epoch} batches={step + 1}/{len(train_loader)}", flush=True)
                z, y = predict(model, select_loader, config["device"], amp)
                omi_index = config["targets"].index("OMI")
                ap = float(average_precision_score(y[:, omi_index], z[:, omi_index]))
                item = {"epoch": epoch, "train_loss": running / len(training), "selection_OMI_AP": ap,
                        "seconds": time.perf_counter() - start}
                history.append(item)
                save_json(out / f"seed_{seed}_history.json", history)
                if ap > best_ap:
                    best_ap = ap
                    torch.save({"state_dict": model.state_dict(), "model": config["model"], "targets": config["targets"],
                                "seed": seed, "epoch": epoch, "selection_OMI_AP": ap, "trainable_parameters": parameters,
                                "manifest_sha256": sha256(data / "manifest.json"),
                                "architecture_sha256": sha256(ROOT / "configs/model_architecture_v1.json")}, out / f"seed_{seed}_best.pt")
                print(json.dumps({"seed": seed, **item}), flush=True)
                save_json(out / "status.json", {**progress, "active_seed": seed, "epoch": epoch})
            progress["completed_seeds"].append(seed)
            save_json(out / f"seed_{seed}_resources.json", {"trainable_parameters": parameters,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if config["device"] == "cuda" else None,
                        "total_seconds": sum(h["seconds"] for h in history)})
        progress["status"] = "trained_awaiting_independent_calibration_and_evaluation"
        save_json(out / "status.json", progress)
    except Exception as error:
        save_json(out / "status.json", {**progress, "status": "failed", "reason": repr(error)})
        raise


def evaluate(config, out, data):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if config != protocol["config"] or sha256(data / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("experiment configuration or data lock differs from training")
    if (out / "metrics.json").exists():
        raise FileExistsError("evaluation already recorded; do not repeatedly evaluate or overwrite")
    state = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if state["completed_seeds"] != config["seeds"]:
        raise ValueError("complete all predeclared seeds before evaluation")
    selection = load_partition(data, "selection")
    calibration = load_partition(data, "calibration")
    evaluation = load_partition(data, "evaluation")
    training = load_partition(data, "train")
    amp = config["amp"] and config["device"] == "cuda"
    result = {"evaluation_status": protocol["evaluation_status"], "seeds": {},
              "working_point_target_selection_sensitivity": config["working_point_sensitivity"],
              "subgroups_status": "descriptive_only_small_subgroups_have_no_inferential_claim",
              "cad_probability": None}
    for seed in config["seeds"]:
        checkpoint = torch.load(out / f"seed_{seed}_best.pt", map_location="cpu", weights_only=True)
        model = make_model(config["model"], config["targets"]).to(config["device"])
        model.load_state_dict(checkpoint["state_dict"])
        zs, ys = predict(model, loader(selection, config), config["device"], amp)
        zc, yc = predict(model, loader(calibration, config), config["device"], amp)
        ze, ye = predict(model, loader(evaluation, config), config["device"], amp)
        reports, calibrators, predictions = {}, {}, []
        for task, target in enumerate(config["targets"]):
            a, b = fit_platt(zc[:, task], yc[:, task])
            threshold = working_threshold(zs[:, task], ys[:, task], config["working_point_sensitivity"])
            prevalence = np.mean([float(r[target]) for r in training])
            probability = expit(a + b * ze[:, task])
            reports[target] = report_target(evaluation, ye[:, task], ze[:, task], probability, threshold,
                                            prevalence, config["bootstrap_replicates"], config["bootstrap_seed"])
            calibrators[target] = {"intercept": float(a), "slope": float(b), "raw_threshold": threshold,
                                  "calibrated_threshold": float(expit(a + b * threshold)),
                                  "selection_sensitivity": float((zs[ys[:, task] == 1, task] >= threshold).mean())}
            for i, row in enumerate(evaluation):
                predictions.append({"record_id": row["record_id"], "patient_id": row["patient_id"], "target": target,
                                    "label": int(ye[i, task]), "raw_logit": float(ze[i, task]),
                                    "probability": float(probability[i]), "screen_positive": bool(ze[i, task] >= threshold)})
        save_json(out / f"seed_{seed}_calibration.json", calibrators)
        save_json(out / f"seed_{seed}_evaluation_predictions.json", predictions)
        result["seeds"][str(seed)] = {"selected_epoch": checkpoint["epoch"], "selection_OMI_AP": checkpoint["selection_OMI_AP"],
                                     "targets": reports}
        save_json(out / "metrics.partial.json", result)
        print(f"Independent evaluation finished for seed {seed}", flush=True)
    save_json(out / "metrics.json", result)
    save_json(out / "status.json", {**state, "status": "calibrated_and_internally_evaluated"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_e0.json")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    data, out = ROOT / config["data_dir"], ROOT / config["out_dir"]
    if args.evaluate:
        evaluate(config, out, data)
    else:
        train(config, args.config, out, data)


if __name__ == "__main__":
    main()
