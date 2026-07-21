"""Hash-validated, atomic, sharded representation cache.

The first implementation appended by loading and rewriting one growing NPY
file. At paper scale that is quadratic in I/O and a Colab interruption between
the vector, row, and manifest writes can misalign examples. This version writes
immutable numbered shards and commits a shard only by atomically replacing the
manifest after all shard files are durable. Orphan files from an interrupted
write are ignored safely.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .utils import stable_hash, write_json

CACHE_TEMPLATE_VERSION = "rv-canonical-2"


def cache_fingerprint(
    family: str,
    checkpoint: str,
    item_split: str,
    k: int,
    option_seed: int,
    template_version: str = CACHE_TEMPLATE_VERSION,
    settings: Mapping[str, Any] | None = None,
) -> str:
    """Identify every setting that can alter a representation.

    ``settings`` should include history selection/retriever, max length,
    pooling, and any model revision. Keeping it optional preserves the compact
    public API used by unit tests while production scripts always provide it.
    """
    normalized = dict(settings or {})
    value = stable_hash(
        family, checkpoint, item_split, int(k), int(option_seed),
        template_version, normalized,
    )
    return f"{value:016x}"


class RepresentationCache:
    def __init__(self, directory: str | Path, fingerprint: str, has_logits: bool):
        self.directory = Path(directory)
        self.fingerprint = str(fingerprint)
        self.has_logits = bool(has_logits)

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        family: str,
        checkpoint: str,
        item_split: str,
        k: int,
        option_seed: int,
        has_logits: bool,
        template_version: str = CACHE_TEMPLATE_VERSION,
        settings: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "RepresentationCache":
        directory = Path(directory)
        fingerprint = cache_fingerprint(
            family, checkpoint, item_split, k, option_seed,
            template_version, settings,
        )
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and not overwrite:
            existing = cls.load(directory)
            if existing.fingerprint != fingerprint:
                raise ValueError(
                    f"Cache {directory} already exists with fingerprint "
                    f"{existing.fingerprint}, expected {fingerprint}; pass overwrite=True explicitly."
                )
            return existing
        if overwrite and directory.exists():
            shutil.rmtree(directory)
        (directory / "shards").mkdir(parents=True, exist_ok=True)
        manifest = {
            "cache_format": 2,
            "fingerprint": fingerprint,
            "family": family,
            "checkpoint": checkpoint,
            "item_split": item_split,
            "k": int(k),
            "option_seed": int(option_seed),
            "template_version": template_version,
            "settings": dict(settings or {}),
            "has_logits": bool(has_logits),
            "n_rows": 0,
            "shards": [],
        }
        write_json(manifest_path, manifest)
        return cls(directory, fingerprint, has_logits)

    def append(
        self,
        rows: pd.DataFrame,
        vectors: np.ndarray,
        logits: np.ndarray | None = None,
    ) -> None:
        """Atomically commit one immutable shard."""
        rows = rows.reset_index(drop=True).copy()
        vectors = np.asarray(vectors, dtype=np.float16)
        if "row_id" not in rows:
            raise ValueError("cache rows require a row_id column")
        row_ids = rows["row_id"].astype(str)
        if row_ids.duplicated().any():
            raise ValueError("duplicate row_id inside cache shard")
        overlap = set(row_ids) & self.already_done_row_ids()
        if overlap:
            raise ValueError(f"cache append would duplicate completed rows: {sorted(overlap)[:3]}")
        if len(rows) != len(vectors):
            raise ValueError(f"rows ({len(rows)}) and vectors ({len(vectors)}) length mismatch")
        if vectors.ndim < 2:
            raise ValueError(f"vectors must have a batch dimension and feature dimension; got {vectors.shape}")
        if self.has_logits != (logits is not None):
            state = "has_logits=True but no logits provided" if self.has_logits else "has_logits=False but logits provided"
            raise ValueError(f"cache declared {state}")
        if logits is not None:
            logits = np.asarray(logits, dtype=np.float32)
            if len(logits) != len(rows):
                raise ValueError("logits length must match rows")

        manifest = self._read_manifest()
        index = len(manifest.get("shards", []))
        prefix = f"{index:06d}"
        shard_dir = self.directory / "shards"
        files = {
            "rows": f"{prefix}.rows.parquet",
            "vectors": f"{prefix}.vectors.npy",
        }
        if logits is not None:
            files["logits"] = f"{prefix}.logits.npy"

        # Temporary files live beside their destination so replace() is atomic.
        temp_rows = shard_dir / f".{files['rows']}.tmp.parquet"
        temp_vectors = shard_dir / f".{files['vectors']}.tmp.npy"
        rows.to_parquet(temp_rows, index=False)
        np.save(temp_vectors, vectors)
        temp_logits = None
        if logits is not None:
            temp_logits = shard_dir / f".{files['logits']}.tmp.npy"
            np.save(temp_logits, logits)
        temp_rows.replace(shard_dir / files["rows"])
        temp_vectors.replace(shard_dir / files["vectors"])
        if temp_logits is not None:
            temp_logits.replace(shard_dir / files["logits"])

        manifest.setdefault("shards", []).append({
            **files,
            "n_rows": int(len(rows)),
            "vector_shape": list(vectors.shape[1:]),
            "logit_shape": list(logits.shape[1:]) if logits is not None else None,
        })
        manifest["n_rows"] = int(sum(shard["n_rows"] for shard in manifest["shards"]))
        write_json(self.directory / "manifest.json", manifest)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expect_family: str | None = None,
        expect_checkpoint: str | None = None,
        expect_item_split: str | None = None,
        expect_k: int | None = None,
        expect_option_seed: int | None = None,
        expect_settings: Mapping[str, Any] | None = None,
    ) -> "RepresentationCache":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No cache manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        checks = {
            "family": expect_family,
            "checkpoint": expect_checkpoint,
            "item_split": expect_item_split,
            "k": expect_k,
            "option_seed": expect_option_seed,
        }
        for name, value in checks.items():
            if value is not None and str(manifest.get(name)) != str(value):
                raise ValueError(
                    f"Cache at {directory} has {name}={manifest.get(name)!r} but caller expects {value!r}"
                )
        if expect_settings is not None and dict(manifest.get("settings", {})) != dict(expect_settings):
            raise ValueError(f"Cache at {directory} has mismatched extraction settings")
        expected_fp = cache_fingerprint(
            str(manifest["family"]), str(manifest["checkpoint"]),
            str(manifest["item_split"]), int(manifest["k"]),
            int(manifest["option_seed"]), str(manifest["template_version"]),
            manifest.get("settings", {}),
        )
        if expected_fp != manifest["fingerprint"]:
            raise ValueError(f"Cache fingerprint mismatch at {directory}: manifest is stale or corrupt")
        cache = cls(directory, manifest["fingerprint"], bool(manifest["has_logits"]))
        cache.validate()
        return cache

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads((self.directory / "manifest.json").read_text())

    def _shards(self) -> list[dict[str, Any]]:
        return list(self._read_manifest().get("shards", []))

    def validate(self) -> dict[str, int]:
        manifest = self._read_manifest()
        total = 0
        for shard in manifest.get("shards", []):
            paths = [self.directory / "shards" / shard["rows"], self.directory / "shards" / shard["vectors"]]
            if self.has_logits:
                paths.append(self.directory / "shards" / shard["logits"])
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise ValueError(f"Committed cache shard is incomplete: {missing}")
            n_rows = len(pd.read_parquet(paths[0], columns=["row_id"]))
            if np.load(paths[1], mmap_mode="r").shape[0] != n_rows:
                raise ValueError(f"row/vector mismatch in {paths[0].name}")
            if self.has_logits and np.load(paths[2], mmap_mode="r").shape[0] != n_rows:
                raise ValueError(f"row/logit mismatch in {paths[0].name}")
            total += n_rows
        if total != int(manifest.get("n_rows", 0)):
            raise ValueError(f"manifest n_rows={manifest.get('n_rows')} but committed shards contain {total}")
        return {"n_rows": total, "n_shards": len(manifest.get("shards", []))}

    def read_rows(self) -> pd.DataFrame:
        shards = self._shards()
        if not shards:
            return pd.DataFrame()
        return pd.concat(
            [pd.read_parquet(self.directory / "shards" / shard["rows"]) for shard in shards],
            ignore_index=True,
        )

    def _read_array(self, key: str) -> np.ndarray:
        shards = self._shards()
        if not shards:
            raise FileNotFoundError(f"cache {self.directory} has no committed {key} shards")
        arrays = [np.load(self.directory / "shards" / shard[key]) for shard in shards]
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

    def read_vectors(self) -> np.ndarray:
        return self._read_array("vectors")

    def read_logits(self) -> np.ndarray:
        if not self.has_logits:
            raise ValueError("this cache has no logits")
        return self._read_array("logits")

    def already_done_row_ids(self) -> set[str]:
        rows = self.read_rows()
        return set() if rows.empty else set(rows["row_id"].astype(str))

    @property
    def n_rows(self) -> int:
        return int(self._read_manifest()["n_rows"])
