#!/usr/bin/env python3
"""Fetch selected public CES files from Harvard Dataverse with checksums.

Only unrestricted files are downloaded. The script queries the dataset API at
run time, records the complete file manifest, verifies the Dataverse MD5, and
adds a SHA-256 fingerprint used by all downstream result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

CES_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    "?persistentId=doi:10.7910/DVN/E9N6PH"
)
FILE_API = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
DEFAULT_FILES = {
    "CES20_Common_OUTPUT_vv.csv",
    "CES20_Common_pre_qx.pdf",
    "CES20_Common_post_qx.pdf",
    "CCES Guide 2020.pdf",
}


def digest(path: Path, name: str) -> str:
    hasher = hashlib.new(name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ResponseVEC-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cross_survey/data/raw/ces2020")
    parser.add_argument("--metadata", default="cross_survey/metadata/ces2020_manifest.json")
    parser.add_argument("--files", nargs="*", default=sorted(DEFAULT_FILES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    metadata_path = Path(args.metadata)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    api_request = urllib.request.Request(
        CES_API, headers={"User-Agent": "ResponseVEC-research/1.0"}
    )
    with urllib.request.urlopen(api_request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API failed: {payload}")

    dataset = payload["data"]
    version = dataset["latestVersion"]
    records = []
    wanted = set(args.files)
    available = {entry["label"]: entry for entry in version["files"]}
    missing = sorted(wanted - set(available))
    if missing:
        raise KeyError(f"requested files absent from Dataverse: {missing}")

    for label in sorted(wanted):
        entry = available[label]
        data = entry["dataFile"]
        if entry.get("restricted", False):
            raise PermissionError(f"refusing restricted file: {label}")
        path = output / label
        expected_md5 = data["checksum"]["value"].lower()
        if args.force or not path.exists() or digest(path, "md5") != expected_md5:
            print(f"downloading {label} ({data['filesize'] / 1024**2:.1f} MiB)")
            download(FILE_API.format(file_id=data["id"]), path)
        actual_md5 = digest(path, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"checksum mismatch for {label}: {actual_md5} != {expected_md5}"
            )
        record = {
            "label": label,
            "file_id": data["id"],
            "bytes": path.stat().st_size,
            "content_type": data["contentType"],
            "source_url": FILE_API.format(file_id=data["id"]),
            "md5": actual_md5,
            "sha256": digest(path, "sha256"),
            "restricted": False,
        }
        records.append(record)
        print(f"verified {label}: sha256={record['sha256'][:16]}...")

    manifest = {
        "dataset": "Cooperative Election Study Common Content, 2020",
        "doi": dataset["persistentUrl"],
        "dataverse_dataset_id": dataset["id"],
        "version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
        "release_time": version["releaseTime"],
        "license": version["license"],
        "api": CES_API,
        "files": records,
    }
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
