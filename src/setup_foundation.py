"""Fetch pinned official sources/base weights, with hashes and safe ZIP paths."""
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile

from ecg_data import ROOT, save_json, sha256
from foundation_ecgfm import config


def download(url, path, headers=None):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-cpc", action="store_true")
    args = parser.parse_args()
    cfg = config()
    archives = [
        ("bowang-lab/ECG-FM", cfg["official_repository_commit"], "ecg-fm-source.zip", "ECG-FM-main"),
        ("Jwoo5/fairseq-signals", cfg["fairseq_signals_commit"], "fairseq-signals-source.zip", "fairseq-signals-master"),
        ("AI4HealthUOL/ecg-fm-benchmarking", "238409835ef55358a10bbc3459dfa9aaa91ad5e5", "ecg-cpc-source.zip", "ecg-fm-benchmarking-main"),
    ]
    provenance = {}
    for repo, commit, filename, directory in archives:
        path = ROOT / "artifacts" / filename
        url = f"https://codeload.github.com/{repo}/zip/{commit}"
        download(url, path)
        with zipfile.ZipFile(path) as archive:
            if archive.comment.decode() != commit:
                raise ValueError(f"source revision differs: {filename}")
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = Path(*Path(member.filename).parts[1:])
                root = ROOT / "external" / directory
                target = root / relative
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("unsafe source archive path")
                data = archive.read(member)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f"official source changed locally: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(data)
        provenance[repo] = {"commit": commit, "url": url, "archive_sha256": sha256(path)}
    checkpoint = ROOT / cfg["checkpoint"]
    download(cfg["checkpoint_url"], checkpoint)
    if sha256(checkpoint) != cfg["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 differs from official LFS object")
    yaml_path = checkpoint.with_suffix(".yaml")
    download(cfg["checkpoint_url"].replace(".pt", ".yaml"), yaml_path)
    save_json(checkpoint.parent / "acquisition.json", {"sources": provenance, "checkpoint_sha256": sha256(checkpoint),
              "checkpoint_url": cfg["checkpoint_url"], "huggingface_revision": cfg["huggingface_revision"],
              "yaml_sha256": sha256(yaml_path), "yaml_matches_checkpoint": False,
              "config_authority": "checkpoint_internal_model_config", "license": cfg["license"]})
    print("Pinned official sources and base checkpoint verified; accompanying YAML mismatch recorded")
    if args.include_cpc:
        import hashlib
        from foundation_ecgcpc import config as cpc_config
        cpc = cpc_config()
        archive_path = ROOT / "artifacts/foundation/ecgcpc_checkpoint.zip"
        download(cpc["checkpoint_url"], archive_path, {"User-Agent": "Mozilla/5.0"})
        if hashlib.md5(archive_path.read_bytes()).hexdigest() != cpc["checkpoint_zip_md5"]:
            raise ValueError("CPC ZIP differs from the official Figshare MD5")
        root = ROOT / "artifacts/foundation/ecgcpc"
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = root / member.filename
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("unsafe CPC checkpoint archive path")
                value = archive.read(member)
                if target.exists() and target.read_bytes() != value:
                    raise ValueError("CPC checkpoint files changed locally")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(value)
        if sha256(ROOT / cpc["checkpoint"]) != cpc["checkpoint_sha256"]:
            raise ValueError("CPC checkpoint SHA256 mismatch")
        save_json(root / "acquisition.json", {"config": cpc, "zip_md5_verified": True,
                  "zip_sha256": sha256(archive_path), "official_source": provenance["AI4HealthUOL/ecg-fm-benchmarking"],
                  "metadata_file": "artifacts/foundation/ecgcpc_article_metadata.json",
                  "serialized_hparams_match_native_yaml": False,
                  "authoritative_configuration": cpc["configuration_authority"]})
        print("CPC checkpoint ZIP and extracted weights verified")


if __name__ == "__main__":
    main()
