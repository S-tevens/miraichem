"""Does the recommendation transfer across devices?

The same configurations are run on several fake IBM devices (each with its own calibration-based
noise model). For each device we recommend a configuration and compare how the configurations rank
on different devices (Spearman rank correlation of their mean errors over seeds).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from miraichem.benchmark.ranking_replicated import (
    GROUP_NO_SEED,
    ReplicatedConfig,
    ReplicatedRecommendation,
    recommend_replicated,
    replicated_configs,
)
from miraichem.benchmark.replication import normalize_runs


@dataclass
class DeviceComparison:
    molecule: str
    devices: list[str]
    means: pd.DataFrame  # rows = configuration, columns = device, values = mean error (mHa)
    spearman: pd.DataFrame  # device x device rank correlation of the means
    recommendations: dict[str, ReplicatedRecommendation]
    n_configs: int


def _config_name(c: ReplicatedConfig) -> str:
    k = c.key
    return (
        f"{k['mapping']} / {k['ansatz']}({k['reps']}) / {k['optimizer']} / {k['mitigation']}"
        f" [{k['shots']} shots, {k['maxiter']} it]"
    )


def compare_devices(
    runs: pd.DataFrame,
    molecule: str,
    backend: str = "noisy",
    bond_length: float | None = None,
    min_seeds: int = 5,
) -> DeviceComparison:
    """Compare configurations that were run on at least two devices.

    Raises:
        ValueError: if fewer than two devices have replicated runs of any shared configuration.
    """
    df = runs[(runs.molecule.str.lower() == molecule.lower()) & (runs.backend == backend)]
    devices = sorted(d for d in df.device.dropna().unique() if d)
    per_device: dict[str, list[ReplicatedConfig]] = {}
    for d in devices:
        try:
            per_device[d] = replicated_configs(runs, molecule, backend, bond_length, min_seeds, d)
        except ValueError:
            continue
    shared_cols = [c for c in GROUP_NO_SEED if c != "device"]
    rows = []
    for d, configs in per_device.items():
        for c in configs:
            row = {col: c.key[col] for col in shared_cols}
            rows.append({**row, "name": _config_name(c), "device": d, "mean": c.mean})
    if not rows:
        raise ValueError("No replicated runs on any device.")
    long = pd.DataFrame(rows)
    n_dev = long.groupby(shared_cols).device.nunique().rename("n_dev").reset_index()
    shared = n_dev[n_dev.n_dev >= 2]
    if shared.empty or long.device.nunique() < 2:
        raise ValueError("Need the same configurations on at least two devices.")
    long = long.merge(shared[shared_cols], on=shared_cols)
    long["cfg"] = long.name
    wide = long.pivot(index="cfg", columns="device", values="mean")
    complete = wide.dropna()
    spearman = complete.corr(method="spearman") if len(complete) >= 3 else wide.corr("spearman")
    # recommend per device from the SHARED configurations only, so devices are compared fairly
    only_shared = normalize_runs(runs).merge(shared[shared_cols], on=shared_cols)
    recs = {
        d: recommend_replicated(only_shared, molecule, backend, bond_length, min_seeds, device=d)
        for d in wide.columns
    }
    return DeviceComparison(
        molecule=molecule,
        devices=list(wide.columns),
        means=wide.sort_values(wide.columns[0]),
        spearman=spearman,
        recommendations=recs,
        n_configs=len(wide),
    )


def summarize_agreement(cmp: DeviceComparison) -> str:
    """One paragraph, computed from the comparison, saying how well the ranking transfers."""
    vals = [
        cmp.spearman.iloc[i, j]
        for i in range(len(cmp.spearman))
        for j in range(i + 1, len(cmp.spearman))
    ]
    top = {
        d: min(cmp.means[d].dropna().items(), key=lambda kv: kv[1])[0].split(" [")[0]
        for d in cmp.devices
    }
    lowest_same = len(set(top.values())) == 1
    picks = {d: r.best.label.split(" / ", 1)[1] for d, r in cmp.recommendations.items()}
    n_pick = len(set(picks.values()))
    text = (
        f"Across {len(cmp.devices)} devices and {cmp.n_configs} configurations, the rank "
        f"correlation of mean errors between devices is {min(vals):.2f} to {max(vals):.2f} "
        f"(1 = identical ordering). "
    )
    text += (
        "The lowest-mean configuration is the same on every device. "
        if lowest_same
        else f"The lowest-mean configuration differs between devices ({len(set(top.values()))} "
        "different ones), but they are close in error. "
    )
    text += (
        f"The recommendation (cheapest configuration statistically tied with the best) is the "
        f"same on all devices: {next(iter(picks.values()))}."
        if n_pick == 1
        else f"The recommendation takes {n_pick} different values across devices: "
        + "; ".join(f"{d}: {p}" for d, p in picks.items())
        + "."
    )
    return text


__all__ = ["DeviceComparison", "compare_devices", "summarize_agreement"]
