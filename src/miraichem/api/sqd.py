"""SQD results for the web app: flat rows, seed-aggregated groups, and the molecules on offer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from miraichem.chemistry.active_space import resolve_active_space
from miraichem.chemistry.molecule import build_mole
from miraichem.config import MoleculeConfig
from miraichem.sqd.problem import FCI_MAX_DETERMINANTS, n_determinants
from miraichem.sqd.result import SqdResult
from miraichem.sqd.storage import load_sqd_results

CHEMICAL_ACCURACY_MHA = 1.6
# Settings that define one study: every seed of the same settings forms one group.
GROUP_FIELDS = (
    "molecule",
    "bond_length",
    "n_reps",
    "shots",
    "readout_noise",
    "iterations",
    "samples_per_batch",
    "num_batches",
)


def sqd_row(r: SqdResult) -> dict[str, Any]:
    c = r.config
    return {
        "hash": r.config_hash,
        "status": r.status,
        "error_message": r.error_message,
        "molecule": c.molecule.name,
        "bond_length": c.bond_length,
        "n_reps": c.n_reps,
        "shots": c.shots,
        "readout_noise": c.readout_noise,
        "iterations": c.iterations,
        "samples_per_batch": c.samples_per_batch,
        "num_batches": c.num_batches,
        "seed": c.seed,
        "qubits": r.num_qubits,
        "e_sqd": r.e_sqd,
        "e_hf": r.e_hf,
        "e_ccsd": r.e_ccsd,
        "e_exact": r.e_exact,
        "error_mha": r.abs_error_mha,
        "hf_error_mha": r.hf_error_mha,
        "ccsd_error_mha": r.ccsd_error_mha,
        "below_hf_mha": r.below_hf_mha,
        "below_ccsd_mha": r.below_ccsd_mha,
        "energy_per_iteration": r.energy_per_iteration,
        "hardware": r.hardware,
        "chem_acc": r.within_chemical_accuracy,
        "n_determinants": r.n_determinants,
        "subspace_dim": r.subspace_dim,
        "subspace_fraction": r.subspace_fraction,
        "distinct_samples": r.n_distinct_samples,
        "postselected": r.n_postselected,
        "error_per_iteration_mha": r.error_per_iteration_mha,
        "wall_time_s": r.wall_time_s,
        "sampling_time_s": r.sampling_time_s,
        "diagonalisation_time_s": r.diagonalisation_time_s,
        "fci_time_s": r.fci_time_s,
        "ccsd_time_s": r.ccsd_time_s,
        "timestamp": r.timestamp,
    }


def sqd_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Seed-aggregate the successful rows that share all settings except the seed."""
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok" or row["error_mha"] is None or row.get("hardware"):
            continue
        buckets.setdefault(tuple(row[f] for f in GROUP_FIELDS), []).append(row)
    groups = []
    for members in buckets.values():
        errs = np.array([m["error_mha"] for m in members], dtype=float)
        first = members[0]
        groups.append(
            {
                **{f: first[f] for f in GROUP_FIELDS},
                "qubits": first["qubits"],
                "n_seeds": len(members),
                "mean_error_mha": float(errs.mean()),
                "std_error_mha": float(errs.std(ddof=1)) if len(errs) > 1 else 0.0,
                "best_error_mha": float(errs.min()),
                "worst_error_mha": float(errs.max()),
                "median_error_mha": float(np.median(errs)),
                "share_chem_acc": float((errs <= CHEMICAL_ACCURACY_MHA).mean()),
                "hf_error_mha": first["hf_error_mha"],
                "ccsd_error_mha": first["ccsd_error_mha"],
                "mean_subspace_fraction": float(
                    np.mean([m["subspace_fraction"] for m in members if m["subspace_fraction"]])
                ),
                "mean_wall_time_s": float(np.mean([m["wall_time_s"] or 0 for m in members])),
                "errors": [float(e) for e in errs],
                "seeds": [m["seed"] for m in members],
            }
        )
    groups.sort(key=lambda g: (g["molecule"], g["mean_error_mha"]))
    return groups


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = [sqd_row(r) for r in load_sqd_results(root)]
    rows.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    return rows


# Simulating the sampling holds a state vector over every determinant; above this the app only
# offers the real-hardware route (the library itself can simulate up to MAX_DETERMINANTS).
UI_SIMULATE_MAX_DETERMINANTS = 2_000_000


@lru_cache(maxsize=16)
def _size(yaml_text: str) -> tuple[int, int, int]:
    """(orbitals, alpha electrons, determinants) of the active space, without running SCF."""
    cfg = MoleculeConfig(**yaml.safe_load(yaml_text))
    mol = build_mole(cfg, 1.0)
    space = resolve_active_space(cfg, mol)
    norb = mol.nao if space is None else space.num_orbitals
    nelec = mol.nelectron if space is None else space.num_electrons
    return norb, nelec // 2, n_determinants(norb, (nelec // 2, nelec // 2))


def sqd_molecules(configs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((Path(configs_dir) / "sqd").glob("*.yaml")):
        text = p.read_text()
        data = yaml.safe_load(text)
        note = " ".join(x.lstrip("#").strip() for x in text.splitlines() if x.startswith("#"))
        norb, nalpha, dets = _size(text)
        out.append(
            {
                "id": p.stem,
                "name": data["name"],
                "basis": data.get("basis", "sto3g"),
                "description": note,
                "qubits": 2 * norb,
                "determinants": dets,
                "simulable": dets <= UI_SIMULATE_MAX_DETERMINANTS,
                "has_exact": dets <= FCI_MAX_DETERMINANTS,
            }
        )
    return out
