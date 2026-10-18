"""Recommend a configuration from seed-replicated runs, with statistics instead of one lucky run.

A single run of a noisy configuration can be lucky. Here runs that differ only in ``seed`` are
grouped, and each configuration gets a mean error, a 95% confidence interval, and the share of
seeds that reached chemical accuracy (with a Wilson interval, which behaves well for small counts).

The recommendation then asks "what can I trust?" in two steps:

1. If some configuration reaches chemical accuracy in at least ``reliability`` of its seeds,
   take the cheapest of those (fewest two-qubit gates, then fewest shots).
2. Otherwise nothing is reliably chemically accurate. Take the configuration with the lowest mean
   error, then the CHEAPEST configuration that is not significantly worse (one-sided Welch t-test,
   p > ``alpha``) and is also practically close (mean within ``margin_factor`` times the best
   mean plus chemical accuracy). The practical margin matters: a very noisy configuration can look
   "not significantly worse" only because its variance is huge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.benchmark.replication import config_label, normalize_runs

CA_MHA = CHEMICAL_ACCURACY_HA * 1000
GROUP_NO_SEED = [
    "molecule",
    "bond_length",
    "mapping",
    "ansatz",
    "reps",
    "optimizer",
    "maxiter",
    "backend",
    "device",
    "shots",
    "mitigation",
    "zne",
]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved for small n and 0% / 100%)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class ReplicatedConfig:
    """One configuration summarised over its seeds."""

    label: str
    key: dict
    errors: list[float]
    gates: float
    shots: float
    n_seeds: int
    mean: float
    std: float
    ci95: float  # half-width of the 95% t interval for the mean
    median: float
    worst: float
    n_chem: int  # seeds within chemical accuracy
    p_chem: float
    p_chem_lo: float
    p_chem_hi: float
    rank: int = 0

    @property
    def cost_key(self) -> tuple[float, float, float]:
        """Cheapest first: two-qubit gates, then shots, then mean error."""
        return (self.gates, self.shots, self.mean)


@dataclass
class ReplicatedRecommendation:
    best: ReplicatedConfig
    lowest_mean: ReplicatedConfig
    tied: list[ReplicatedConfig]
    ranked: list[ReplicatedConfig]
    reliable: bool
    justification: str
    params: dict = field(default_factory=dict)


def _mean_or_zero(series: pd.Series) -> float:
    value = series.mean()
    return 0.0 if pd.isna(value) else float(value)


def replicated_configs(
    runs: pd.DataFrame,
    molecule: str,
    backend: str,
    bond_length: float | None = None,
    min_seeds: int = 3,
    device: str | None = None,
) -> list[ReplicatedConfig]:
    """Summarise every configuration of one molecule/backend that has at least ``min_seeds`` seeds.

    ``runs`` is the per-run table from :func:`miraichem.benchmark.replication.runs_frame`.
    Without ``bond_length`` the geometry with the most runs is used. ``device`` keeps only runs on
    that fake IBM device (default: all devices, each device being a separate configuration).
    """
    df = normalize_runs(runs)
    df = df[(df.molecule.str.lower() == molecule.lower()) & (df.backend == backend)]
    if device is not None:
        df = df[df.device == device]
    if df.empty:
        raise ValueError(f"No runs for molecule={molecule!r}, backend={backend!r}.")
    if bond_length is None:
        bond_length = float(df.bond_length.value_counts().sort_index().idxmax())
    df = df[np.isclose(df.bond_length, bond_length)]
    out: list[ReplicatedConfig] = []
    for key, g in df.groupby(GROUP_NO_SEED):
        g = g.drop_duplicates("seed")
        n = len(g)
        if n < min_seeds:
            continue
        e = g.error_mHa.to_numpy(dtype=float)
        std = float(e.std(ddof=1))
        n_chem = int((e < CA_MHA).sum())
        lo, hi = wilson_interval(n_chem, n)
        gates = _mean_or_zero(g.two_qubit_gates) or _mean_or_zero(g.logical_depth)
        keyd = dict(zip(GROUP_NO_SEED, key, strict=True))
        out.append(
            ReplicatedConfig(
                label=config_label(keyd),
                key=keyd,
                errors=[float(x) for x in e],
                gates=gates,
                shots=_mean_or_zero(g.total_shots),
                n_seeds=n,
                mean=float(e.mean()),
                std=std,
                ci95=float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)),
                median=float(np.median(e)),
                worst=float(e.max()),
                n_chem=n_chem,
                p_chem=n_chem / n,
                p_chem_lo=lo,
                p_chem_hi=hi,
            )
        )
    out.sort(key=lambda c: (c.mean, c.cost_key))
    for i, c in enumerate(out, start=1):
        c.rank = i
    return out


def not_worse_than(
    cand: ReplicatedConfig,
    best: ReplicatedConfig,
    alpha: float = 0.05,
    margin_factor: float = 1.5,
) -> bool:
    """True if ``cand`` is statistically and practically as good as ``best``.

    Statistically: a one-sided Welch t-test cannot show that cand's mean error is higher (p >
    alpha). Practically: its mean is within ``margin_factor * best.mean + 1.6 mHa``.
    """
    if cand is best:
        return True
    if cand.mean > margin_factor * best.mean + CA_MHA:
        return False
    a, b = np.array(cand.errors), np.array(best.errors)
    if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
        return cand.mean <= best.mean + 1e-9
    p = stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue
    return bool(p > alpha)


def recommend_replicated(
    runs: pd.DataFrame,
    molecule: str,
    backend: str,
    bond_length: float | None = None,
    min_seeds: int = 3,
    reliability: float = 0.5,
    alpha: float = 0.05,
    margin_factor: float = 1.5,
    device: str | None = None,
) -> ReplicatedRecommendation:
    """Recommend the cheapest configuration you can trust, using all seeds (see module docs).

    Raises:
        ValueError: if no configuration has at least ``min_seeds`` seeds.
    """
    ranked = replicated_configs(runs, molecule, backend, bond_length, min_seeds, device)
    if not ranked:
        raise ValueError(f"No configuration has {min_seeds}+ seeds for {molecule}/{backend}.")
    lowest = ranked[0]
    params = {"reliability": reliability, "alpha": alpha, "margin_factor": margin_factor}
    reliable_pool = [c for c in ranked if c.p_chem >= reliability]
    if reliable_pool:
        tied = reliable_pool
        reliable = True
    else:
        tied = [c for c in ranked if not_worse_than(c, lowest, alpha, margin_factor)]
        reliable = False
    best = min(tied, key=lambda c: c.cost_key)
    return ReplicatedRecommendation(
        best=best,
        lowest_mean=lowest,
        tied=tied,
        ranked=ranked,
        reliable=reliable,
        justification=_justify(best, lowest, tied, ranked, reliable, molecule, backend, params),
        params=params,
    )


def _gate_text(c: ReplicatedConfig) -> str:
    if not c.gates:
        return "no two-qubit gates"
    return f"{c.gates:.0f} two-qubit gate" + ("" if round(c.gates) == 1 else "s")


def _justify(best, lowest, tied, ranked, reliable, molecule, backend, params) -> str:
    n = best.n_seeds
    parts = [
        f"Recommended for {molecule} on the {backend} backend (bond length "
        f"{best.key['bond_length']} A): {best.label}.",
        f"Over {n} seeds its mean error is {best.mean:.1f} ± {best.std:.1f} mHa "
        f"(95% interval for the mean ±{best.ci95:.1f}; median {best.median:.1f}, worst "
        f"{best.worst:.1f}) and it reached chemical accuracy (1.6 mHa) in {best.n_chem} of {n} "
        f"seeds (95% interval {best.p_chem_lo:.0%} to {best.p_chem_hi:.0%}). It uses "
        f"{_gate_text(best)} and {best.shots:,.0f} shots.",
    ]
    if reliable:
        parts.append(
            f"It is the cheapest of the {len(tied)} configurations that reach chemical accuracy "
            f"in at least {params['reliability']:.0%} of seeds."
        )
    else:
        parts.append(
            f"None of the {len(ranked)} configurations reaches chemical accuracy in "
            f"{params['reliability']:.0%} of seeds, so none is reliably chemically accurate."
        )
        if best is lowest:
            parts.append("It has the lowest mean error of all configurations.")
        else:
            parts.append(
                f"The lowest mean error is {lowest.mean:.1f} mHa ({lowest.label}); the "
                f"recommendation is the cheapest of the {len(tied)} configurations whose mean "
                "error is not significantly higher (one-sided Welch t-test, "
                f"p > {params['alpha']}) and is within {params['margin_factor']}x that mean plus "
                "1.6 mHa."
            )
    if min(c.n_seeds for c in tied) < 5:
        parts.append("Some tied configurations have fewer than 5 seeds, so treat ties with care.")
    return " ".join(parts)


def replicated_dataframe(ranked: list[ReplicatedConfig]) -> pd.DataFrame:
    """Leaderboard of replicated configurations as a DataFrame (dashboard and CLI)."""
    return pd.DataFrame(
        [
            {
                "rank": c.rank,
                "configuration": c.label,
                "seeds": c.n_seeds,
                "mean_mHa": c.mean,
                "std_mHa": c.std,
                "ci95_mHa": c.ci95,
                "median_mHa": c.median,
                "worst_mHa": c.worst,
                "within_1.6mHa": f"{c.n_chem}/{c.n_seeds}",
                "p_chem_low": c.p_chem_lo,
                "p_chem_high": c.p_chem_hi,
                "two_qubit_gates": c.gates,
                "total_shots": c.shots,
            }
            for c in ranked
        ]
    )
