"""Per-method compute cost accounting (design §6.3, §9.3).

Wraps each method's extraction + prediction with wall-clock and peak-memory
instrumentation so the accuracy--cost Pareto frontier (Figure 2) is grounded in
measured GPU-hours per 1000 respondents, not estimates.

A CostRecord is produced once per (method, regime, K) and aggregated into a
cost table by ``cost_table``. The measurement contract:

  * ``gpu_hours_per_1k`` — forward-pass + head time, scaled to 1000 respondents.
  * ``latency_ms_per_respondent`` — per-respondent wall clock (excludes one-time
    model load, which is amortized and reported separately).
  * ``peak_memory_gib`` — peak GPU memory during extraction (0.0 on CPU).
  * ``n_forward_passes`` — number of LLM forward passes (0 for analytic
    baselines like majority/MIRT-prior).

Timing excludes model loading, dataset parsing, and metric computation, so the
cost table reflects the marginal cost of imputing one more respondent — the
quantity a practitioner budgets for.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class CostRecord:
    method: str
    regime: str               # "R1" | "R2" | "R3"
    k: int
    n_respondents: int
    n_forward_passes: int
    elapsed_seconds: float
    peak_memory_gib: float = 0.0

    @property
    def gpu_hours_per_1k(self) -> float:
        if self.n_respondents <= 0:
            return 0.0
        return self.elapsed_seconds * (1000.0 / self.n_respondents) / 3600.0

    @property
    def latency_ms_per_respondent(self) -> float:
        if self.n_respondents <= 0:
            return 0.0
        return self.elapsed_seconds * 1000.0 / self.n_respondents

    def to_row(self) -> dict:
        return {
            "method": self.method, "regime": self.regime, "k": int(self.k),
            "n_respondents": int(self.n_respondents),
            "n_forward_passes": int(self.n_forward_passes),
            "elapsed_seconds": float(self.elapsed_seconds),
            "gpu_hours_per_1k": float(self.gpu_hours_per_1k),
            "latency_ms_per_respondent": float(self.latency_ms_per_respondent),
            "peak_memory_gib": float(self.peak_memory_gib),
        }


@contextmanager
def timed_extraction(method: str, regime: str, k: int, n_respondents: int,
                     n_forward_passes: int):
    """Context manager that yields a CostRecord being filled. Peak GPU memory
    is sampled on entry/exit if torch+cuda are available; otherwise 0.0."""
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    record = CostRecord(
        method=method, regime=regime, k=int(k), n_respondents=int(n_respondents),
        n_forward_passes=int(n_forward_passes), elapsed_seconds=0.0,
    )
    try:
        yield record
    finally:
        torch_sync = None
        try:
            if cuda:
                torch.cuda.synchronize()
                torch_sync = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        except Exception:
            torch_sync = None
        record.elapsed_seconds = float(time.perf_counter() - start)
        if torch_sync is not None:
            record.peak_memory_gib = float(torch_sync)


def cost_table(records: Iterable[CostRecord]):
    import pandas as pd  # delayed so the module imports without pandas (CPU dev)
    return pd.DataFrame([r.to_row() for r in records])


def save_cost_table(records: Iterable[CostRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = cost_table(records)
    table.to_csv(output_path, index=False)
    return output_path


def mixed_allocation_cost(m1_cost: float, m2_cost: float, fraction_m1: float) -> float:
    """M3 mixed-allocation GPU-hours per 1000: a convex combination of M1 and
    M2 costs weighted by the fraction of items routed to ResponseVec. The
    fraction is estimated on a calibration set (see tradeoff.tex §4.4)."""
    if not 0.0 <= fraction_m1 <= 1.0:
        raise ValueError("fraction_m1 must be in [0, 1]")
    return float(fraction_m1) * float(m1_cost) + (1.0 - float(fraction_m1)) * float(m2_cost)
