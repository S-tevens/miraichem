"""JSON views of the results for the web app: seed-replicated studies with recommendations, the
cross-device comparison, hardware evaluations and dissociation curves.

Every number comes from saved runs; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from miraichem.analysis.dissociation import DissociationCurve
from miraichem.benchmark.devices import compare_devices, summarize_agreement
from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.benchmark.ranking_replicated import (
    ReplicatedConfig,
    recommend_replicated,
    replicated_configs,
)
from miraichem.benchmark.replication import normalize_runs
from miraichem.benchmark.report import load_hardware_results
from miraichem.benchmark.storage import load_all_results

CA_MHA = CHEMICAL_ACCURACY_HA * 1000
MIN_SEEDS = 5


def clean(value: Any) -> Any:
    """Make a value JSON-safe: NaN/inf become None, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def config_json(c: ReplicatedConfig) -> dict[str, Any]:
    k = c.key
    return {
        "label": c.label,
        "mapping": k["mapping"],
        "ansatz": k["ansatz"],
        "reps": int(k["reps"]),
        "optimizer": k["optimizer"],
        "maxiter": int(k["maxiter"]),
        "shots": int(k["shots"]),
        "mitigation": k["mitigation"],
        "device": k["device"],
        "bond_length": float(k["bond_length"]),
        "errors": c.errors,
        "n_seeds": c.n_seeds,
        "mean": c.mean,
        "std": c.std,
        "ci95": c.ci95,
        "median": c.median,
        "worst": c.worst,
        "best": min(c.errors),
        "n_chem": c.n_chem,
        "p_chem": c.p_chem,
        "p_chem_lo": c.p_chem_lo,
        "p_chem_hi": c.p_chem_hi,
        "two_qubit_gates": c.gates,
        "total_shots": c.shots,
        "rank": c.rank,
    }


def recommendation_json(rec) -> dict[str, Any]:
    return {
        "label": rec.best.label,
        "reliable": rec.reliable,
        "justification": rec.justification,
        "lowest_mean_label": rec.lowest_mean.label,
        "tied_labels": [c.label for c in rec.tied],
        "params": rec.params,
    }


def build_studies(runs: pd.DataFrame, hf_error: dict[str, float]) -> list[dict[str, Any]]:
    studies = []
    runs = normalize_runs(runs)
    for (molecule, backend), g in runs.groupby(["molecule", "backend"]):
        devices = sorted(d for d in g.device.unique() if d) or [""]
        for device in devices:
            try:
                configs = replicated_configs(
                    runs, str(molecule), str(backend), min_seeds=MIN_SEEDS, device=device or None
                )
            except ValueError:
                continue
            if not configs:
                continue
            try:
                rec = recommend_replicated(
                    runs, str(molecule), str(backend), min_seeds=MIN_SEEDS, device=device or None
                )
                recommendation = recommendation_json(rec)
            except ValueError:
                recommendation = None
            studies.append(
                {
                    "id": f"{molecule}-{backend}-{device or 'exact'}".lower(),
                    "molecule": str(molecule),
                    "backend": str(backend),
                    "device": device,
                    "bond_length": configs[0].key["bond_length"],
                    "hf_error_mha": hf_error.get(str(molecule)),
                    "configs": [config_json(c) for c in configs],
                    "recommendation": recommendation,
                }
            )
    return studies


def build_devices(runs: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for molecule in sorted(runs.molecule.unique()):
        try:
            cmp = compare_devices(runs, str(molecule), "noisy", min_seeds=MIN_SEEDS)
        except ValueError:
            continue
        out.append(
            {
                "molecule": str(molecule),
                "devices": cmp.devices,
                "configs": [
                    {"name": str(name), "means": {d: row[d] for d in cmp.devices}}
                    for name, row in cmp.means.iterrows()
                ],
                "spearman": {
                    a: {b: cmp.spearman.loc[a, b] for b in cmp.spearman.columns}
                    for a in cmp.spearman.index
                },
                "recommendations": {
                    d: {
                        "label": r.best.label,
                        "mean": r.best.mean,
                        "std": r.best.std,
                        "n_chem": r.best.n_chem,
                        "n_seeds": r.best.n_seeds,
                    }
                    for d, r in cmp.recommendations.items()
                },
                "summary": summarize_agreement(cmp),
            }
        )
    return out


def build_hardware(hardware_dir: Path, run_labels: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for h in load_hardware_results(hardware_dir):
        ideal = abs(h.e_ideal_at_params - h.e_exact) * 1000
        sim = abs(h.e_simulator - h.e_exact) * 1000
        out.append(
            {
                "source": h.source_config_hash,
                "source_label": run_labels.get(h.source_config_hash, ""),
                "molecule": h.molecule,
                "bond_length": h.bond_length,
                "device": h.backend,
                "shots": h.shots,
                "timestamp": h.timestamp,
                "hf_error_mha": abs(h.e_hf - h.e_exact) * 1000,
                "ideal_error_mha": ideal,
                "simulator_error_mha": sim,
                "simulator_mitigation": h.simulator_mitigation,
                "evaluations": [
                    {
                        "mitigation": e.mitigation,
                        "error_mha": e.abs_error_mha,
                        "std_mha": e.e_hw_std * 1000,
                        "job_id": e.job_id,
                    }
                    for e in h.evaluations
                ],
            }
        )
    return out


def build_curves(curves_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(curves_dir).glob("*.json")):
        c = DissociationCurve(**json.loads(path.read_text()))
        out.append(
            {
                "molecule": c.molecule,
                "label": c.label,
                "backend": c.backend,
                "bond_lengths": c.bond_lengths,
                "e_hf": c.e_hf,
                "e_exact": c.e_exact,
                "e_vqe": c.e_vqe,
                "error_mha": c.error_mha,
                "hf_error_mha": c.hf_error_mha,
            }
        )
    return out


def hf_errors(results_dir: Path) -> dict[str, float]:
    """Hartree-Fock error (mHa) per molecule at its most common geometry."""
    out: dict[str, float] = {}
    for r in load_all_results(results_dir):
        if r.status != "ok" or r.e_hf is None or r.e_exact is None:
            continue
        name = r.config.molecule.name
        if name not in out:
            out[name] = abs(r.e_hf - r.e_exact) * 1000
    return out
