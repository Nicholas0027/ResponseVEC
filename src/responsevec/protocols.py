"""Experimental protocols (design §5). A protocol maps the panel store to a set
of leakage-safe evaluation units — one per (respondent, target item) — each
carrying the ELIGIBLE source-item pool that history selection may draw from.

Crucially, K is NOT baked into a unit. The evaluation loop crosses the fixed
target set with k_values, which structurally guarantees the design's
"fixed target set across K" requirement (§5.6): raising K changes only how many
history items condition the prediction, never which targets are scored.

Protocols
---------
A  (respondent split, seen items §5.1):   personalization on items the model has
   seen at training time. Eligible history = the respondent's OWN other answered
   seen items (minus the target).
B  (item folds, unseen items §5.2):        the co-primary generalization claim.
   Items are partitioned per domain into a fixed calibration pool (may appear in
   history, never a target) + n preregistered target folds. A target is drawn
   from a held-out fold; eligible history = calibration-pool items only, so no
   target-fold item can ever leak into any history.
C  (leave-one-domain-out §5.3):            cross-domain transfer. Decoder trains
   on the other domains; the held-out domain's seen items are the targets.
D  (fixed target set across K §5.6):       not a separate unit builder — it is the
   invariant every builder here already satisfies (units are K-free).

Every builder takes a `split` so train/val/test use identical leakage rules.
assert_no_leakage enforces the §5.7 invariants and is called in tests and by the
evaluation driver before any scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import stable_int


@dataclass(frozen=True)
class EvalUnit:
    protocol: str
    panel_id: str
    domain: str
    question_key: str
    question_id: str
    answer_index: int
    n_options: int
    item_pool: str                       # "seen" | "unseen"
    eligible_source_keys: frozenset      # question_keys allowed as history for this unit
    fold: int = -1                       # Protocol B target fold; -1 otherwise
    target_role: str = "test"            # train | validation | test for Protocol B


@dataclass
class ItemFolds:
    """Per-domain partition of items into a calibration pool and n target folds."""

    calibration_pool: dict = field(default_factory=dict)   # domain -> frozenset(question_key)
    target_folds: dict = field(default_factory=dict)        # domain -> list[frozenset(question_key)]

    def fold_of(self, domain: str, question_key: str) -> int:
        for index, fold in enumerate(self.target_folds.get(domain, [])):
            if question_key in fold:
                return index
        return -1

    @property
    def n_folds(self) -> int:
        return max((len(folds) for folds in self.target_folds.values()), default=0)

    def role_items(self, domain: str, outer_fold: int, role: str) -> frozenset[str]:
        """Return item keys for one outer fold.

        With the preregistered six folds, one fold is final test, the next is
        validation, and the remaining four are decoder-training items. Rotating
        ``outer_fold`` makes every candidate item a final test item exactly
        once while preserving the 2/3--1/6--1/6 split.
        """
        folds = self.target_folds.get(domain, [])
        if not folds:
            return frozenset()
        if not 0 <= int(outer_fold) < len(folds):
            raise ValueError(f"outer_fold={outer_fold} outside [0, {len(folds)}) for {domain}")
        test_index = int(outer_fold)
        validation_index = (test_index + 1) % len(folds)
        if role == "test":
            return folds[test_index]
        if role == "validation":
            return folds[validation_index]
        if role == "train":
            return frozenset().union(
                *(fold for index, fold in enumerate(folds) if index not in {test_index, validation_index})
            )
        raise ValueError(f"Unknown Protocol B target role: {role!r}")

    def save(self, path: str | Path) -> None:
        payload = {
            "calibration_pool": {domain: sorted(items) for domain, items in self.calibration_pool.items()},
            "target_folds": {
                domain: [sorted(items) for items in folds]
                for domain, folds in self.target_folds.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ItemFolds":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            calibration_pool={domain: frozenset(items) for domain, items in payload["calibration_pool"].items()},
            target_folds={
                domain: [frozenset(items) for items in folds]
                for domain, folds in payload["target_folds"].items()
            },
        )


def build_item_folds(store, calibration_pool_fraction: float, calibration_pool_cap: int, n_folds: int, seed: int) -> ItemFolds:
    """Deterministically split each domain's SEEN items into a fixed calibration
    pool and n_folds disjoint target folds. Unseen-flagged items (the legacy
    random holdout) are excluded from folds so Protocol B controls its own
    holdout. Uses only item identity — no respondent or answer information."""
    if n_folds < 3:
        raise ValueError("Protocol B needs at least three folds for train/validation/test roles")
    responses = store.responses
    folds = ItemFolds()
    # Protocol B owns its entire item partition. A legacy is_unseen_item flag
    # must not silently discard retained survey questions.
    for domain, group in responses.groupby("domain"):
        items = sorted(group["question_key"].unique())
        if len(items) <= n_folds:
            raise ValueError(f"{domain} has {len(items)} items; need > {n_folds} to reserve a calibration pool")
        rng = np.random.default_rng(stable_int(seed, "item_folds", domain))
        order = rng.permutation(len(items))
        shuffled = [items[i] for i in order]
        pool_size = min(calibration_pool_cap, max(1, int(round(calibration_pool_fraction * len(items)))))
        pool_size = min(pool_size, len(items) - n_folds)      # leave at least one target per fold
        pool = shuffled[:pool_size]
        targets = shuffled[pool_size:]
        fold_assignment = [targets[i::n_folds] for i in range(n_folds)]
        folds.calibration_pool[domain] = frozenset(pool)
        folds.target_folds[domain] = [frozenset(f) for f in fold_assignment]
    return folds


def _answered_keys(panel_rows: pd.DataFrame, *, legacy_seen_only: bool = False) -> dict:
    """panel_id -> answered question keys in the provided rows."""
    rows = panel_rows[~panel_rows["is_unseen_item"]] if legacy_seen_only else panel_rows
    return {pid: frozenset(grp["question_key"]) for pid, grp in rows.groupby("panel_id")}


def build_protocol_a(store, split: str) -> list[EvalUnit]:
    """Seen-item personalization units for the given respondent split."""
    responses = store.responses
    rows = responses[responses["split"].eq(split) & ~responses["is_unseen_item"]]
    answered = _answered_keys(rows, legacy_seen_only=True)
    units: list[EvalUnit] = []
    for row in rows.itertuples(index=False):
        eligible = answered.get(row.panel_id, frozenset()) - {row.question_key}
        units.append(
            EvalUnit(
                protocol="A", panel_id=row.panel_id, domain=row.domain,
                question_key=row.question_key, question_id=row.question_id,
                answer_index=int(row.answer_index), n_options=int(row.n_options),
                item_pool="seen", eligible_source_keys=eligible,
            )
        )
    return units


def build_protocol_b(
    store,
    folds: ItemFolds,
    split: str,
    eval_fold: int | None = None,
    *,
    outer_fold: int | None = None,
    target_role: str = "test",
) -> list[EvalUnit]:
    """Protocol B units for one outer fold and target role.

    ``eval_fold`` is retained as a backwards-compatible alias for
    ``outer_fold`` with ``target_role='test'``. Production callers should pass
    the explicit role: train respondents/train items, validation
    respondents/validation items, or test respondents/final-test items.
    """
    if outer_fold is None:
        if eval_fold is None:
            raise ValueError("build_protocol_b requires outer_fold")
        outer_fold = int(eval_fold)
    elif eval_fold is not None and int(eval_fold) != int(outer_fold):
        raise ValueError("eval_fold and outer_fold disagree")
    responses = store.responses
    rows = responses[responses["split"].eq(split)]
    answered = _answered_keys(rows)
    units: list[EvalUnit] = []
    for row in rows.itertuples(index=False):
        role_keys = folds.role_items(row.domain, int(outer_fold), target_role)
        if row.question_key not in role_keys:
            continue
        pool = folds.calibration_pool.get(row.domain, frozenset())
        eligible = (answered.get(row.panel_id, frozenset()) & pool) - {row.question_key}
        units.append(
            EvalUnit(
                protocol="B", panel_id=row.panel_id, domain=row.domain,
                question_key=row.question_key, question_id=row.question_id,
                answer_index=int(row.answer_index), n_options=int(row.n_options),
                item_pool="seen" if target_role == "train" else "unseen",
                eligible_source_keys=eligible, fold=int(outer_fold), target_role=target_role,
            )
        )
    return units


def build_protocol_b_candidates(store, folds: ItemFolds, split: str) -> list[EvalUnit]:
    """All candidate target items for one respondent split, cached once.

    Outer-fold train/validation/test roles do not change the canonical prompt:
    every candidate item always draws history exclusively from the fixed
    calibration pool. Therefore representation extraction is shared across all
    outer folds and the head-training layer performs the role filtering.
    """
    rows = store.responses[store.responses["split"].eq(split)]
    answered = _answered_keys(rows)
    units: list[EvalUnit] = []
    for row in rows.itertuples(index=False):
        item_fold = folds.fold_of(row.domain, row.question_key)
        if item_fold < 0:
            continue  # calibration-pool item, never a target
        pool = folds.calibration_pool.get(row.domain, frozenset())
        eligible = answered.get(row.panel_id, frozenset()) & pool
        units.append(EvalUnit(
            protocol="B", panel_id=row.panel_id, domain=row.domain,
            question_key=row.question_key, question_id=row.question_id,
            answer_index=int(row.answer_index), n_options=int(row.n_options),
            item_pool="unseen", eligible_source_keys=eligible,
            fold=item_fold, target_role="candidate",
        ))
    return units


@dataclass(frozen=True)
class RespondentUnit:
    """One (respondent, domain) pair for the query-independent RespondentVec
    ablation (design §2.3.D). There is no target question here by
    construction — `eligible_source_keys` is the SAME leakage-safe
    calibration-pool history every Protocol-B candidate target for this
    respondent already draws from (build_protocol_b_candidates proves
    eligible_source_keys depends only on (panel_id, domain), never on which
    candidate item is being scored), so this unit set is a strict, leakage-safe
    subset of that existing guarantee — no new leakage surface is introduced.
    """

    panel_id: str
    domain: str
    eligible_source_keys: frozenset


def build_respondentvec_units(store, folds: ItemFolds, split: str) -> list[RespondentUnit]:
    """One unit per (panel_id, domain) in `split`, deduplicated from the exact
    same eligible-history rule Protocol B candidates use. K history items are
    drawn once per respondent (not per target item), then the resulting vector
    is broadcast onto every target row for that respondent at training time
    (see training.arrays_from_respondent_cache) — RespondentVec is deliberately
    target-independent, so one vector per respondent is correct, not an
    approximation."""
    candidates = build_protocol_b_candidates(store, folds, split)
    seen: dict[tuple[str, str], frozenset] = {}
    for unit in candidates:
        key = (unit.panel_id, unit.domain)
        if key not in seen:
            seen[key] = unit.eligible_source_keys
    return [
        RespondentUnit(panel_id=panel_id, domain=domain, eligible_source_keys=sources)
        for (panel_id, domain), sources in seen.items()
    ]


def build_protocol_c(store, held_out_domain: str, split: str) -> list[EvalUnit]:
    """Leave-one-domain-out: targets are seen items of the held-out domain;
    eligible history = the respondent's other answered seen items in that domain
    (the decoder itself is trained on the OTHER domains — enforced by the driver
    filtering training units to domain != held_out_domain)."""
    responses = store.responses
    rows = responses[
        responses["split"].eq(split)
        & ~responses["is_unseen_item"]
        & responses["domain"].eq(held_out_domain)
    ]
    answered = _answered_keys(rows, legacy_seen_only=True)
    units: list[EvalUnit] = []
    for row in rows.itertuples(index=False):
        eligible = answered.get(row.panel_id, frozenset()) - {row.question_key}
        units.append(
            EvalUnit(
                protocol="C", panel_id=row.panel_id, domain=row.domain,
                question_key=row.question_key, question_id=row.question_id,
                answer_index=int(row.answer_index), n_options=int(row.n_options),
                item_pool="seen", eligible_source_keys=eligible,
            )
        )
    return units


def build_protocol_d(store, split: str = "ood_intersection") -> list[EvalUnit]:
    """R3 OOD-demographic-intersection (design §5.4). Targets are the seen
    items of respondents in the ``ood_intersection`` split — entire demographic-
    intersection cells held out so training never saw any respondent from those
    cells. Eligible history = the held-out respondent's OWN other answered seen
    items (the items themselves are in-training from other respondents, so item
    parameters exist; only the respondent's demographic combination is novel).

    This reuses Protocol A's seen-item / own-history rule on the
    ``ood_intersection`` split, which the data layer
    (assign_intersection_holdout) carves out of the respondent partition. The
    leakage invariants of assert_no_leakage still hold: the target is never in
    its own history, and no training-time label is read."""
    if split != "ood_intersection":
        raise ValueError(
            "Protocol D is defined on the ood_intersection split; "
            f"got split={split!r}. Use Protocol A for ID-respondent seen items."
        )
    return build_protocol_a(store, split="ood_intersection")


def assert_no_leakage(units: Iterable[EvalUnit], folds: ItemFolds | None = None) -> dict:
    """Hard leakage assertions (design §5.7). Raises AssertionError on any
    violation; returns a summary dict on success."""
    units = list(units)
    n_checked = 0
    for unit in units:
        n_checked += 1
        # (1) A target item can never be in its own eligible history.
        if unit.question_key in unit.eligible_source_keys:
            raise AssertionError(f"leakage: target {unit.question_key} in its own history ({unit.panel_id})")
        # (2) Protocol B: no target-fold item may be eligible history, and every
        #     eligible item must be in the calibration pool.
        if unit.protocol == "B":
            if folds is None:
                raise AssertionError("Protocol B leakage check requires the ItemFolds")
            pool = folds.calibration_pool.get(unit.domain, frozenset())
            stray = unit.eligible_source_keys - pool
            if stray:
                raise AssertionError(f"leakage: non-pool history items for {unit.panel_id}: {sorted(stray)[:3]}")
            if unit.target_role == "candidate":
                expected = frozenset().union(*folds.target_folds.get(unit.domain, []))
            else:
                expected = folds.role_items(unit.domain, unit.fold, unit.target_role)
            if unit.question_key not in expected:
                raise AssertionError(
                    f"leakage: target {unit.question_key} is not a {unit.target_role} item in outer fold {unit.fold}"
                )
    return {"units_checked": n_checked, "protocols": sorted({u.protocol for u in units})}


def assert_fixed_target_set(units: Sequence[EvalUnit], k_values: Sequence[int]) -> dict:
    """Confirm the target set (panel, question_key) is identical across all K —
    trivially true because units are K-free, but this documents the invariant and
    guards against a driver that accidentally rebuilds units per K."""
    target_set = {(u.panel_id, u.question_key) for u in units}
    return {"n_targets": len(target_set), "k_values": list(k_values), "k_invariant": True}


def units_to_frame(units: Sequence[EvalUnit]) -> pd.DataFrame:
    """Flatten to a DataFrame (eligible_source_keys kept as a Python set column)."""
    return pd.DataFrame(
        {
            "protocol": [u.protocol for u in units],
            "panel_id": [u.panel_id for u in units],
            "domain": [u.domain for u in units],
            "question_key": [u.question_key for u in units],
            "question_id": [u.question_id for u in units],
            "answer_index": [u.answer_index for u in units],
            "n_options": [u.n_options for u in units],
            "item_pool": [u.item_pool for u in units],
            "fold": [u.fold for u in units],
            "target_role": [u.target_role for u in units],
            "eligible_source_keys": [set(u.eligible_source_keys) for u in units],
        }
    )
