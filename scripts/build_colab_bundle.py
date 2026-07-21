#!/usr/bin/env python
"""Create the self-contained source bundle consumed by the Colab notebook."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "artifacts", "tmp", ".venv",
    "responsevec.egg-info",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/responsevec_colab.zip")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = (project / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project.rglob("*")):
            if not path.is_file() or path == output:
                continue
            relative = path.relative_to(project)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if relative.parts and relative.parts[0] == "output" and relative.suffix not in {".ipynb"}:
                continue
            archive.write(path, Path("responsevec") / relative)
    print({"bundle": str(output), "size_mb": round(output.stat().st_size / 2**20, 2)})


if __name__ == "__main__":
    main()
