"""Rank benchmark runs, find Pareto-optimal ones, and recommend a configuration.

Three costs compete in this project: energy ERROR (how wrong), two-qubit GATE count (how deep, so
how noisy) and total SHOTS (how much quantum time). No configuration is best at everything, so we
report (1) a weighted score for a single ordering, (2) Pareto fronts that show every sensible
trade-off, and (3) a recommendation: the cheapest configuration that still reaches chemical
accuracy (error < 1.6 mHa), or the best-scoring one if none does.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from miraichem.vqe.result import RunResult

ERROR_FLOOR_MHA = 1e-3  # errors below this count as "exact" (log scale needs a floor)


@dataclass(frozen=True)
class Weights:
    """Importance of each term in the score (they are normalised to sum to 1).

    Defaults follow the project brief: accuracy weighs most, then two-qubit gates, then shots.
    """

    accuracy: float = 0.7
    gates: float = 0.2
    shots: float = 0.1

    def __post_init__(self) -> None:
        if min(self.accuracy, self.gates, self.shots) < 0 or self.total <= 0:
            raise ValueError("Weights must be non-negative and not all zero.")

    @property
    def total(self) -> float:
        return self.accuracy + self.gates + self.shots

    def normalized(self) -> Weights:
        t = self.total
        return Weights(self.accuracy / t, self.gates / t, self.shots / t)


@dataclass
class RankedRun:
    rank: int
    score: float  # lower is better, in [0, 1]
    cost: float  # gates + shots part only (accuracy ignored), in [0, 1]; lower is cheaper
    result: RunResult
    error_mha: float
    gates: float
    shots: int
    within_chemical_accuracy: bool
    on_front_shots: bool = False  # on the (error, shots) Pareto front
    on_front_gates: bool = False  # on the (error, gates) Pareto front

    @property
    def label(self) -> str:
        return describe(self.result)


@dataclass
class Recommendation:
    best: RankedRun
    justification: str
    ranked: list[RankedRun]
    n_candidates: int
    reached_chemical_accuracy: bool
    pareto_shots: list[RankedRun] = field(default_factory=list)
    pareto_gates: list[RankedRun] = field(default_factory=list)


def describe(result: RunResult) -> str:
    """Short human-readable name of a run's configuration."""
    c = result.config
    text = f"{c.ansatz}(reps={c.ansatz_reps})/{c.optimizer}, {c.mapping}"
    if c.mitigation != "none":
        text += f", {c.mitigation}"
    return text


def gate_cost(result: RunResult) -> float:
    """Two-qubit gate count after transpilation; for ideal runs (no transpilation) the logical
    circuit depth is used as the complexity proxy. Ranking is per backend, so units never mix."""
    if result.two_qubit_gates is not None:
        return float(result.two_qubit_gates)
    return float(result.logical_depth or 0)


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated by any other (both coordinates are minimised).

    A point is dominated if another is no worse in both and strictly better in at least one.
    Exact duplicates do not dominate each other, so both stay on the front.
    """
    front = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            xj <= xi and yj <= yi and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(points)
            if j != i
        )
        if not dominated:
            front.append(i)
    return front


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _usable(results: list[RunResult]) -> list[RunResult]:
    return [r for r in results if r.status == "ok" and r.abs_error_mha is not None]


def rank_results(results: list[RunResult], weights: Weights | None = None) -> list[RankedRun]:
    """Order runs by weighted score (best first) and mark Pareto-front membership.

    Score = w_acc * norm(log10 error) + w_gates * norm(gates) + w_shots * norm(shots), each term
    min-max normalised over the runs being compared, so 0 is best and 1 is worst. Error is on a
    log scale because accuracy spans orders of magnitude (0.001 to 1000 mHa).
    """
    runs = _usable(results)
    if not runs:
        return []
    w = (weights or Weights()).normalized()
    errors = [max(r.abs_error_mha or 0.0, ERROR_FLOOR_MHA) for r in runs]
    gates = [gate_cost(r) for r in runs]
    shots = [r.total_shots or 0 for r in runs]
    n_err = _minmax([math.log10(e) for e in errors])
    n_gates, n_shots = _minmax(gates), _minmax([float(s) for s in shots])

    cost_weight = w.gates + w.shots
    ranked = [
        RankedRun(
            rank=0,
            score=w.accuracy * n_err[i] + w.gates * n_gates[i] + w.shots * n_shots[i],
            cost=(w.gates * n_gates[i] + w.shots * n_shots[i]) / cost_weight
            if cost_weight
            else 0.0,
            result=r,
            error_mha=errors[i],
            gates=gates[i],
            shots=shots[i],
            within_chemical_accuracy=bool(r.within_chemical_accuracy),
        )
        for i, r in enumerate(runs)
    ]
    for i in pareto_front([(r.error_mha, float(r.shots)) for r in ranked]):
        ranked[i].on_front_shots = True
    for i in pareto_front([(r.error_mha, r.gates) for r in ranked]):
        ranked[i].on_front_gates = True
    ranked.sort(key=lambda r: (r.score, r.error_mha, r.result.config_hash))
    for position, run in enumerate(ranked, start=1):
        run.rank = position
    return ranked


def select_runs(
    results: list[RunResult], molecule: str, backend: str, bond_length: float | None = None
) -> list[RunResult]:
    """Successful runs for one molecule and backend at one bond length.

    If ``bond_length`` is omitted and several are present, the one with the most runs is used
    (smallest on ties), so a single-geometry benchmark just works.
    """
    runs = [
        r
        for r in _usable(results)
        if r.config.molecule.name.lower() == molecule.lower() and r.config.backend == backend
    ]
    if not runs:
        raise ValueError(
            f"No successful runs found for molecule={molecule!r}, backend={backend!r}."
        )
    if bond_length is None:
        counts = Counter(r.config.bond_length for r in runs)
        bond_length = min(counts, key=lambda b: (-counts[b], b))
    runs = [r for r in runs if math.isclose(r.config.bond_length, bond_length, abs_tol=1e-9)]
    if not runs:
        raise ValueError(f"No runs at bond length {bond_length} A.")
    return runs


def _justify(
    best: RankedRun, ranked: list[RankedRun], reached: bool, molecule: str, backend: str
) -> str:
    r = best.result
    pool = [x for x in ranked if x.within_chemical_accuracy] if reached else ranked
    lowest_error = min(ranked, key=lambda x: (x.error_mha, x.rank))
    hf_mha = abs((r.e_hf or 0) - (r.e_exact or 0)) * 1000
    parts = [
        f"Recommended for {molecule} on the {backend} backend (bond length "
        f"{r.config.bond_length} A): {best.label}.",
        f"Its energy error is {best.error_mha:.3g} mHa (Hartree-Fock alone is off by "
        f"{hf_mha:.1f} mHa), using "
        + (
            f"{best.gates:.0f} two-qubit gates"
            if r.two_qubit_gates is not None
            else f"a logical circuit depth of {best.gates:.0f}"
        )
        + f" and {best.shots:,} shots.",
    ]
    if reached:
        parts.append(
            f"It is the cheapest (fewest gates and shots) among the {len(pool)} of {len(ranked)} "
            f"runs that reach chemical accuracy (error below 1.6 mHa)."
        )
    else:
        parts.append(
            f"None of the {len(ranked)} runs reached chemical accuracy (1.6 mHa); this is the "
            f"best-scoring one, so treat the result as the least bad option, not as accurate."
        )
    if lowest_error is not best:
        extras = []
        if best.shots and lowest_error.shots > best.shots:
            extras.append(f"{lowest_error.shots / best.shots:.1f}x the shots")
        if best.gates and lowest_error.gates > best.gates:
            extras.append(f"{lowest_error.gates / best.gates:.1f}x the gates")
        parts.append(
            f"The lowest-error run ({lowest_error.label}, {lowest_error.error_mha:.3g} mHa)"
            + (f" needs {' and '.join(extras)}." if extras else " ranks lower on cost.")
        )
    fronts = [
        name
        for name, on in (
            ("error vs shots", best.on_front_shots),
            ("error vs two-qubit gates", best.on_front_gates),
        )
        if on
    ]
    if fronts:
        parts.append(f"It lies on the Pareto front for {' and '.join(fronts)}.")
    return " ".join(parts)


def recommend(
    results: list[RunResult],
    molecule: str,
    backend: str,
    weights: Weights | None = None,
    bond_length: float | None = None,
) -> Recommendation:
    """Recommend a configuration: the cheapest one reaching chemical accuracy, else the best score.

    Raises:
        ValueError: if there are no successful runs for this molecule and backend.
    """
    runs = select_runs(results, molecule, backend, bond_length)
    ranked = rank_results(runs, weights)
    reached = any(x.within_chemical_accuracy for x in ranked)
    if reached:
        # Once chemical accuracy is reached, extra accuracy is worth nothing: take the cheapest.
        accurate = [x for x in ranked if x.within_chemical_accuracy]
        best = min(accurate, key=lambda x: (x.cost, x.error_mha, x.result.config_hash))
    else:
        best = ranked[0]
    return Recommendation(
        best=best,
        justification=_justify(best, ranked, reached, molecule, backend),
        ranked=ranked,
        n_candidates=len(ranked),
        reached_chemical_accuracy=reached,
        pareto_shots=[x for x in ranked if x.on_front_shots],
        pareto_gates=[x for x in ranked if x.on_front_gates],
    )


def ranking_dataframe(ranked: list[RankedRun]):
    """Leaderboard as a pandas DataFrame (used by the dashboard and reports)."""
    import pandas as pd

    rows = []
    for x in ranked:
        c = x.result.config
        rows.append(
            {
                "rank": x.rank,
                "score": round(x.score, 4),
                "ansatz": c.ansatz,
                "reps": c.ansatz_reps,
                "optimizer": c.optimizer,
                "mapping": c.mapping,
                "mitigation": c.mitigation,
                "shots_per_group": c.shots if c.backend != "ideal" else None,
                "error_mHa": x.error_mha,
                "chemical_accuracy": x.within_chemical_accuracy,
                "two_qubit_gates": x.result.two_qubit_gates,
                "logical_depth": x.result.logical_depth,
                "total_shots": x.shots,
                "n_evals": x.result.n_function_evals,
                "pareto_shots": x.on_front_shots,
                "pareto_gates": x.on_front_gates,
                "config_hash": x.result.config_hash,
            }
        )
    return pd.DataFrame(rows)
