"""Summarise replicated runs: the same configuration run with several seeds.

A single seed can be lucky or unlucky (shot noise, random start, SPSA perturbations). Grouping runs
that differ only in ``seed`` gives mean, spread and success rate, which is what "reliable" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.vqe.result import RunResult

GROUP_COLS = [
    "molecule",
    "bond_length",
    "mapping",
    "ansatz",
    "reps",
    "optimizer",
    "maxiter",
    "backend",
    "shots",
    "mitigation",
]


RUN_COLUMNS = [*GROUP_COLS, "seed", "error_mHa", "n_evals"]
RUNS_CSV = "replication_runs.csv"
TABLE_CSV = "replication_table.csv"


def runs_frame(results: list[RunResult]) -> pd.DataFrame:
    """One row per successful run, with the columns needed to group runs that differ by seed."""
    rows = []
    for r in results:
        if r.status != "ok" or r.abs_error_mha is None:
            continue
        c = r.config
        rows.append(
            {
                "molecule": c.molecule.name,
                "bond_length": c.bond_length,
                "mapping": c.mapping,
                "ansatz": c.ansatz,
                "reps": c.ansatz_reps,
                "optimizer": c.optimizer,
                "maxiter": c.maxiter,
                "backend": c.backend,
                "shots": c.shots if c.backend != "ideal" else 0,
                "mitigation": c.mitigation,
                "seed": c.seed,
                "error_mHa": r.abs_error_mha,
                "n_evals": r.n_function_evals,
            }
        )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def replication_table(
    results: list[RunResult], molecule: str | None = None, min_seeds: int = 3
) -> pd.DataFrame:
    """One row per configuration with at least ``min_seeds`` distinct seeds.

    Columns: n_seeds, mean/std/median/min/max error (mHa), a 95% interval for the mean
    (t-distribution), and the fraction of seeds within chemical accuracy. Sorted by mean error.
    """
    return summarize(runs_frame(results), molecule, min_seeds)


def summarize(df: pd.DataFrame, molecule: str | None = None, min_seeds: int = 3) -> pd.DataFrame:
    """Same as :func:`replication_table`, from a per-run table (see :func:`runs_frame`)."""
    if df.empty:
        return df
    if molecule:
        df = df[df.molecule.str.lower() == molecule.lower()]
    rows = []
    for key, g in df.groupby(GROUP_COLS):
        g = g.drop_duplicates("seed")
        n = len(g)
        if n < min_seeds:
            continue
        e = g.error_mHa.to_numpy()
        std = float(e.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
        rows.append(
            dict(
                zip(GROUP_COLS, key, strict=True),
                n_seeds=n,
                mean_mHa=float(e.mean()),
                std_mHa=std,
                ci95_mHa=half,
                median_mHa=float(np.median(e)),
                min_mHa=float(e.min()),
                max_mHa=float(e.max()),
                frac_chem_acc=float((e < CHEMICAL_ACCURACY_HA * 1000).mean()),
                mean_evals=float(g.n_evals.mean()),
            )
        )
    out = pd.DataFrame(rows)
    return out.sort_values("mean_mHa").reset_index(drop=True) if not out.empty else out


def config_label(row) -> str:
    """Short readable name, e.g. ``parity / uccsd(1) / spsa / readout+zne``.

    ``row`` may be a pandas row, a namedtuple from ``itertuples`` or a dict.
    """

    def get(key: str):
        return row[key] if isinstance(row, dict | pd.Series) else getattr(row, key)

    return (
        f"{get('mapping')} / {get('ansatz')}({get('reps')}) / "
        f"{get('optimizer')} / {get('mitigation')}"
    )


def read_runs_csv(path) -> pd.DataFrame:
    """Per-run replication table saved by ``miraichem replicate --export-runs``."""
    return pd.read_csv(path)
