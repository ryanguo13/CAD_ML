"""Versioned waveform index, patient isolation, and on-demand mV loading."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import wfdb

from baseline_ecg import LEADS, TARGETS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/\u65b0\u5efa\u6587\u4ef6\u5939")
SPLITS = ("train", "selection", "calibration", "evaluation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty table: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_waveform(path):
    """Decode WFDB gain/baseline; return canonical 12 x 5000 physical mV."""
    record = wfdb.rdrecord(str(Path(path).with_suffix("")), physical=False)
    if record.n_sig != 12 or record.fs != 500 or record.sig_len != 5000:
        raise ValueError("expected 12 leads, 500 Hz, 5000 samples")
    if len(set(record.sig_name)) != 12 or set(record.sig_name) != set(LEADS):
        raise ValueError("missing, repeated, or unsupported leads")
    if any(unit != "mV" for unit in record.units):
        raise ValueError("unverified voltage units")
    if any(fmt != "16" for fmt in record.fmt):
        raise ValueError("only audited WFDB format 16 is enabled")
    gain = np.asarray(record.adc_gain, dtype=np.float64)
    if not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("invalid ADC gain")
    digital = record.d_signal
    if digital.shape != (5000, 12) or (digital == -32768).any():
        raise ValueError("missing digital samples or invalid waveform length")
    x = (digital - np.asarray(record.baseline)) / gain
    x = x[:, [record.sig_name.index(lead) for lead in LEADS]].T.astype(np.float32)
    if not np.isfinite(x).all() or (np.ptp(x, axis=1) == 0).any():
        raise ValueError("nonfinite samples or flat lead")
    if (digital == 32767).any():
        raise ValueError("ADC rail clipping")
    return x


def quality_flags(x):
    # Engineering screening flags, not validated clinical QC diagnoses.
    flags = []
    if np.abs(x).max() > 10:
        flags.append("amplitude_over_10mv")
    if np.ptp(x.reshape(12, 10, 500).mean(2), axis=1).max() > 0.5:
        flags.append("baseline_drift_candidate")
    spectrum = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True))) ** 2
    freq = np.fft.rfftfreq(5000, d=1 / 500)
    power = spectrum[:, (freq >= 0.5) & (freq <= 100)].sum(1)
    line = spectrum[:, (freq >= 49) & (freq <= 51)].sum(1)
    if (line / np.maximum(power, 1e-12) > 0.2).any():
        flags.append("50hz_noise_candidate")
    return flags


def grouped_split(rows, seed=20261010):
    """Stratify patient groups on any OMI/AMI, linking exact byte duplicates."""
    parent = {r["patient_id"]: r["patient_id"] for r in rows}

    def find(patient):
        while parent[patient] != patient:
            parent[patient] = parent[parent[patient]]
            patient = parent[patient]
        return patient

    owner = {}
    for row in rows:
        key = row["waveform_sha256"]
        if key in owner:
            a, b = find(row["patient_id"]), find(owner[key])
            parent[max(a, b)] = min(a, b)
        else:
            owner[key] = row["patient_id"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[find(row["patient_id"])].append(row)
    strata = defaultdict(list)
    for group, members in grouped.items():
        key = tuple(max(int(row[target]) for row in members) for target in ("OMI", "AMI"))
        strata[key].append(group)
    rng = np.random.default_rng(seed)
    assignment = {}
    for key in sorted(strata):
        groups = np.asarray(sorted(strata[key]))
        rng.shuffle(groups)
        boundaries = [0, round(len(groups) * .6), round(len(groups) * .7), round(len(groups) * .8), len(groups)]
        for split, lo, hi in zip(SPLITS, boundaries[:-1], boundaries[1:]):
            assignment.update({group: split for group in groups[lo:hi]})
    return {patient: assignment[find(patient)] for patient in parent}


def verify_manifest(data_dir):
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["file_sha256"].items():
        if sha256(Path(data_dir) / filename) != expected:
            raise ValueError(f"locked index modified: {filename}")
    rows = read_csv(Path(data_dir) / "records.csv")
    patients, hashes = {}, {}
    for row in rows:
        for registry, key in ((patients, row["patient_id"]), (hashes, row["waveform_sha256"])):
            if key in registry and registry[key] != row["split"]:
                raise ValueError("patient or exact waveform crosses splits")
            registry[key] = row["split"]
    return manifest


def prepare(args):
    if args.out.exists():
        raise FileExistsError("choose a new version directory; existing data locks are immutable")
    raw = read_csv(args.csv)
    hidden = read_csv(args.hidden_csv)
    if {r["Patient_id"] for r in raw} & {r["Patient_id"] for r in hidden}:
        raise ValueError("public labeled/hidden patient overlap")
    records, labels, excluded, joined = [], [], [], []
    seen = set()
    for split_source, source_rows in (("labeled", raw), ("hidden", hidden)):
        for i, row in enumerate(source_rows):
            record_id = "angiography:" + Path(row["ecg_row_record"]).stem
            path = args.wave_dir / row["ecg_row_record"]
            if record_id in seen or path.parent.resolve() != args.wave_dir.resolve():
                raise ValueError("repeated record id or unsafe waveform path")
            seen.add(record_id)
            try:
                x = read_waveform(path)
                flags = quality_flags(x)
                wave_hash, header_hash = sha256(path), sha256(path.with_suffix(".hea"))
            except (ValueError, OSError, IndexError) as error:
                excluded.append({"record_id": record_id, "source": split_source, "reason": str(error)})
                continue
            rec = dict(dataset_id="angiography_figshare_29925314", dataset_version="local_2026-10-10",
                       patient_id="angiography:" + row["Patient_id"], encounter_id="", record_id=record_id,
                       waveform_path=str(path.resolve()), lead_names="|".join(LEADS), fs=500, duration=10,
                       units="mV", quality_flags="|".join(flags), waveform_sha256=wave_hash,
                       header_sha256=header_hash, split="hidden" if split_source == "hidden" else "")
            records.append(rec)
            lab = dict(record_id=record_id, label_source="original_train_csv" if split_source == "labeled" else "unreleased",
                       label_definition_version="original_dataset_historical_diagnoses_v1")
            for target in TARGETS:
                if split_source == "hidden":
                    lab[target] = ""
                elif target == "ACS":
                    lab[target] = int(row["AMI"] == "1" or row["UA"] == "1")
                else:
                    if row[target] not in ("0", "1"):
                        raise ValueError(f"invalid label: {record_id}/{target}")
                    lab[target] = int(row[target])
            lab["ami_subtype_disagreement"] = "" if split_source == "hidden" else int(
                int(row["AMI"]) != int(row["STEMI"] == "1" or row["NSTEMI"] == "1"))
            lab.update({field: row[field] for field in ("age", "gender", "Time_Interval")})
            lab.update({field: row.get(field, "") for field in ("Prior_PCI", "Paced", "VF_VT")})
            labels.append(lab)
            if split_source == "labeled":
                joined.append({**rec, **lab})
            if (i + 1) % 4000 == 0:
                print(f"Audited {split_source}: {i + 1}/{len(source_rows)}", flush=True)
    assignment = grouped_split(joined, args.seed)
    for rec in records:
        if rec["split"] != "hidden":
            rec["split"] = assignment[rec["patient_id"]]
    hidden_hashes = {r["waveform_sha256"] for r in records if r["split"] == "hidden"}
    if hidden_hashes & {r["waveform_sha256"] for r in records if r["split"] != "hidden"}:
        raise ValueError("exact waveform duplicates overlap public hidden holdout")
    args.out.mkdir(parents=True)
    write_csv(args.out / "records.csv", records)
    write_csv(args.out / "labels.csv", labels)
    write_csv(args.out / "patients.csv", [dict(patient_id=p, split=s) for p, s in sorted(assignment.items())])
    save_json(args.out / "excluded_records.json", excluded)
    by_id = {row["record_id"]: row for row in labels}
    summary = {}
    for split in (*SPLITS, "hidden"):
        selected = [r for r in records if r["split"] == split]
        summary[split] = {"records": len(selected), "patients": len({r["patient_id"] for r in selected}),
                          "positive": {t: sum(int(by_id[r["record_id"]][t]) for r in selected)
                                       for t in TARGETS} if split != "hidden" else None}
    save_json(args.out / "manifest.json", {
        "protocol_version": "1.0", "seed": args.seed, "split_method": "patient_and_exact_duplicate_groups_stratified_any_OMI_AMI_60_10_10_20",
        "evaluation_status": "internal_previously_explored_source_not_external_validation",
        "raw_sources": {str(p.resolve()): sha256(p) for p in (args.csv, args.hidden_csv)},
        "file_sha256": {f: sha256(args.out / f) for f in ("records.csv", "labels.csv", "patients.csv", "excluded_records.json")},
        "splits": summary, "excluded_records": len(excluded),
        "exact_duplicate_records": len(records) - len({r["waveform_sha256"] for r in records}),
        "label_disagreements_retained": sum(int(r["ami_subtype_disagreement"] or 0) for r in labels),
        "qc_flags": dict(Counter(flag for r in records for flag in r["quality_flags"].split("|") if flag)),
        "qc_status": "engineering_flags_only_not_clinically_validated",
        "near_duplicate_detection": "pending", "cad_reference_available": False,
    })
    verify_manifest(args.out)
    print(json.dumps(summary, indent=2), flush=True)


def load_partition(data_dir, split):
    if split not in SPLITS:
        raise ValueError("hidden labels are unavailable and cannot be used for training/evaluation")
    verify_manifest(data_dir)
    labels = {r["record_id"]: r for r in read_csv(Path(data_dir) / "labels.csv")}
    return [{**r, **labels[r["record_id"]]} for r in read_csv(Path(data_dir) / "records.csv") if r["split"] == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=SOURCE / "CSV/train.csv")
    ap.add_argument("--hidden-csv", type=Path, default=SOURCE / "CSV/test.csv")
    ap.add_argument("--wave-dir", type=Path, default=SOURCE / "row_data")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/data_v1")
    ap.add_argument("--seed", type=int, default=20261010)
    prepare(ap.parse_args())


if __name__ == "__main__":
    main()
