"""Ranking across a bond-length scan.

Ranking at one geometry can mislead: a shallow ansatz may hit chemical accuracy at equilibrium yet
fail badly when the bond is stretched, where electron correlation is strongest. Here a
"configuration" is everything except the bond length, and it is judged on the WORST error over all
scanned bond lengths and on coverage (the fraction of bond lengths within chemical accuracy).
Only configurations run at several bond lengths take part (e.g. produced by `miraichem curve`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from miraichem.benchmark.ranking import (
    ERROR_FLOOR_MHA,
    Weights,
    _minmax,
    describe,
    gate_cost,
)
from miraichem.vqe.result import RunResult


@dataclass
class ScanSummary:
    """One configuration's behaviour across a bond-length scan."""

    key: str  # hash of the configuration with the bond length neutralised
    result: RunResult  # representative run (for labels and config details)
    bond_lengths: list[float]
    n_attempted: int
    n_ok: int
    worst_error_mha: float
    mean_error_mha: float
    coverage: float  # fraction of ATTEMPTED bond lengths within chemical accuracy
    gates: float  # mean gate cost over the scan
    shots: float  # mean total shots over the scan
    score: float = 0.0
    cost: float = 0.0
    rank: int = 0

    @property
    def label(self) -> str:
        return describe(self.result)

    @property
    def reaches_everywhere(self) -> bool:
        return self.n_ok == self.n_attempted and self.coverage == 1.0


@dataclass
class ScanRecommendation:
    best: ScanSummary
    justification: str
    ranked: list[ScanSummary]
    n_configs: int
    reached_chemical_accuracy_everywhere: bool


def scan_key(result: RunResult) -> str:
    """Identity of a configuration independent of bond length (same hash for every point)."""
    return result.config.model_copy(update={"bond_length": 1.0}).config_hash()


def summarize_scans(
    results: list[RunResult], molecule: str, backend: str, min_points: int = 3
) -> list[ScanSummary]:
    """Group runs of one molecule/backend by configuration; keep scans with enough points."""
    groups: dict[str, list[RunResult]] = {}
    for r in results:
        if r.config.molecule.name.lower() == molecule.lower() and r.config.backend == backend:
            groups.setdefault(scan_key(r), []).append(r)
    out = []
    for key, runs in groups.items():
        ok = [r for r in runs if r.status == "ok" and r.abs_error_mha is not None]
        lengths = sorted({r.config.bond_length for r in runs})
        if len(lengths) < min_points or not ok:
            continue
        errors = [max(r.abs_error_mha or 0.0, ERROR_FLOOR_MHA) for r in ok]
        out.append(
            ScanSummary(
                key=key,
                result=ok[0],
                bond_lengths=lengths,
                n_attempted=len(runs),
                n_ok=len(ok),
                worst_error_mha=max(errors),
                mean_error_mha=sum(errors) / len(errors),
                coverage=sum(bool(r.within_chemical_accuracy) for r in ok) / len(runs),
                gates=sum(gate_cost(r) for r in ok) / len(ok),
                shots=sum(r.total_shots or 0 for r in ok) / len(ok),
            )
        )
    return out


def rank_scans(scans: list[ScanSummary], weights: Weights | None = None) -> list[ScanSummary]:
    """Order scans by weighted score on (worst error, gates, shots); best first."""
    if not scans:
        return []
    w = (weights or Weights()).normalized()
    n_err = _minmax([math.log10(s.worst_error_mha) for s in scans])
    n_gates, n_shots = _minmax([s.gates for s in scans]), _minmax([s.shots for s in scans])
    cost_weight = w.gates + w.shots
    for i, s in enumerate(scans):
        s.score = w.accuracy * n_err[i] + w.gates * n_gates[i] + w.shots * n_shots[i]
        s.cost = (w.gates * n_gates[i] + w.shots * n_shots[i]) / cost_weight if cost_weight else 0.0
    ranked = sorted(scans, key=lambda s: (s.score, s.worst_error_mha, s.key))
    for position, s in enumerate(ranked, start=1):
        s.rank = position
    return ranked


def recommend_across_scan(
    results: list[RunResult],
    molecule: str,
    backend: str,
    weights: Weights | None = None,
    min_points: int = 3,
) -> ScanRecommendation:
    """Recommend the cheapest configuration that is chemically accurate at EVERY scanned bond
    length; if none is, the one with the best coverage (then best score).

    Raises:
        ValueError: if no configuration was run at ``min_points`` or more bond lengths.
    """
    scans = summarize_scans(results, molecule, backend, min_points)
    if not scans:
        raise ValueError(
            f"No configuration for molecule={molecule!r}, backend={backend!r} was run at "
            f"{min_points}+ bond lengths. Create scans with `miraichem curve` first."
        )
    ranked = rank_scans(scans, weights)
    everywhere = [s for s in ranked if s.reaches_everywhere]
    if everywhere:
        best = min(everywhere, key=lambda s: (s.cost, s.worst_error_mha, s.key))
    else:
        best = max(ranked, key=lambda s: (s.coverage, -s.score, s.key))
    return ScanRecommendation(
        best=best,
        justification=_justify_scan(best, ranked, bool(everywhere), molecule, backend),
        ranked=ranked,
        n_configs=len(ranked),
        reached_chemical_accuracy_everywhere=bool(everywhere),
    )


def _justify_scan(
    best: ScanSummary, ranked: list[ScanSummary], everywhere: bool, molecule: str, backend: str
) -> str:
    lo, hi = min(best.bond_lengths), max(best.bond_lengths)
    unit = "two-qubit gates" if best.result.two_qubit_gates is not None else "logical depth"
    parts = [
        f"Recommended for {molecule} on the {backend} backend across a scan of "
        f"{len(best.bond_lengths)} bond lengths ({lo} to {hi} A): {best.label}.",
        f"Its worst-case error is {best.worst_error_mha:.3g} mHa (mean "
        f"{best.mean_error_mha:.3g} mHa), it is within chemical accuracy (1.6 mHa) at "
        f"{best.coverage:.0%} of the bond lengths, and it uses on average {best.gates:.0f} "
        f"{unit} and {best.shots:,.0f} shots per point.",
    ]
    if everywhere:
        n = sum(s.reaches_everywhere for s in ranked)
        if n == 1:
            parts.append(
                f"It is the only one of {len(ranked)} scanned configurations that stays "
                f"chemically accurate at every bond length."
            )
        else:
            parts.append(
                f"It is the cheapest of the {n} (of {len(ranked)}) scanned configurations that "
                f"stay chemically accurate at every bond length."
            )
    else:
        if len(ranked) == 1:
            lead = "The only scanned configuration does not stay"
        else:
            lead = f"None of the {len(ranked)} scanned configurations stays"
        parts.append(
            f"{lead} chemically accurate at every bond length; this one has the best coverage, "
            f"so treat it as the least bad option."
        )
    worse = [s for s in ranked if s is not best and s.worst_error_mha > best.worst_error_mha]
    if worse:
        w = max(worse, key=lambda s: s.worst_error_mha)
        parts.append(
            f"For contrast, {w.label} is within chemical accuracy at {w.coverage:.0%} of the bond "
            f"lengths with a worst-case error of {w.worst_error_mha:.3g} mHa."
        )
    return " ".join(parts)


def scan_dataframe(ranked: list[ScanSummary]):
    """Scan leaderboard as a pandas DataFrame (for the dashboard and reports)."""
    import pandas as pd

    rows = []
    for s in ranked:
        c = s.result.config
        rows.append(
            {
                "rank": s.rank,
                "score": round(s.score, 4),
                "ansatz": c.ansatz,
                "reps": c.ansatz_reps,
                "optimizer": c.optimizer,
                "mapping": c.mapping,
                "mitigation": c.mitigation,
                "points": len(s.bond_lengths),
                "worst_error_mHa": s.worst_error_mha,
                "mean_error_mHa": s.mean_error_mha,
                "coverage": s.coverage,
                "mean_gates": s.gates,
                "mean_shots": s.shots,
                "scan_key": s.key,
            }
        )
    return pd.DataFrame(rows)
