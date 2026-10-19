"""Data access for the web app: loads saved results per source and serialises them to plain dicts.

Two sources are offered: ``published`` (the committed results) and ``local`` (runs made on this
machine, including those started from the app). Loaded results are cached and refreshed when files
change, so the API stays fast with a thousand runs.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from miraichem.analysis.dissociation import default_bond_lengths
from miraichem.api.views import (
    build_curves,
    build_devices,
    build_hardware,
    build_studies,
    clean,
    hf_errors,
)
from miraichem.benchmark.ranking import RankedRun, describe
from miraichem.benchmark.replication import RUNS_CSV, normalize_runs, read_runs_csv, runs_frame
from miraichem.benchmark.storage import NON_RUN_DIRS, load_all_results, load_result
from miraichem.chemistry.classical import compute_reference
from miraichem.config import MoleculeConfig
from miraichem.vqe.result import RunResult

SOURCE_LABELS = {"published": "Published results", "local": "My runs"}
EQUILIBRIUM = {"h2": 0.735, "lih": 1.595}
DEVICES = ["FakeSherbrooke", "FakeBrisbane", "FakeKyiv", "FakeFez", "FakeTorino"]

OPTIONS: dict[str, Any] = {
    "mappings": [
        {"id": "parity", "label": "Parity", "hint": "fewer qubits (two-qubit reduction)"},
        {"id": "jordan_wigner", "label": "Jordan-Wigner", "hint": "one qubit per spin orbital"},
    ],
    "ansatzes": [
        {"id": "uccsd", "label": "UCCSD", "hint": "chemistry-inspired, deep"},
        {"id": "hea", "label": "Hardware-efficient", "hint": "shallow rotations and CNOTs"},
        {
            "id": "hea_hf",
            "label": "Hardware-efficient, HF start",
            "hint": "starts exactly at Hartree-Fock",
        },
    ],
    "optimizers": [
        {"id": "cobyla", "label": "COBYLA", "hint": "derivative-free; stalls on shot noise"},
        {"id": "spsa", "label": "SPSA", "hint": "noise-tolerant"},
        {"id": "spsa_clip", "label": "SPSA, clipped", "hint": "SPSA that stays near a good start"},
        {"id": "lbfgsb", "label": "L-BFGS-B", "hint": "exact values only (ideal backend)"},
    ],
    "backends": [
        {"id": "ideal", "label": "Ideal simulator", "hint": "exact statevector, no noise"},
        {"id": "noisy", "label": "Noisy simulator", "hint": "noise model of a real IBM device"},
    ],
    "mitigations": [
        {"id": "none", "label": "None"},
        {"id": "readout", "label": "Readout correction"},
        {"id": "zne", "label": "Zero-noise extrapolation"},
        {"id": "readout+zne", "label": "Readout + ZNE"},
    ],
    "devices": [{"id": d, "label": d.replace("Fake", "")} for d in DEVICES],
}


def run_row(r: RunResult) -> dict[str, Any]:
    """One flat row per run, for tables and filters."""
    c = r.config
    return {
        "hash": r.config_hash,
        "status": r.status,
        "molecule": c.molecule.name,
        "bond_length": c.bond_length,
        "mapping": c.mapping,
        "ansatz": c.ansatz,
        "reps": c.ansatz_reps,
        "optimizer": c.optimizer,
        "maxiter": c.maxiter,
        "backend": c.backend,
        "device": c.fake_backend_name if c.backend != "ideal" else "",
        "shots": c.shots if c.backend != "ideal" else 0,
        "mitigation": c.mitigation,
        "seed": c.seed,
        "error_mha": r.abs_error_mha,
        "chem_acc": r.within_chemical_accuracy,
        "e_vqe": r.e_vqe,
        "e_exact": r.e_exact,
        "e_hf": r.e_hf,
        "hf_error_mha": None
        if r.e_hf is None or r.e_exact is None
        else abs(r.e_hf - r.e_exact) * 1000,
        "qubits": r.num_qubits,
        "params": r.num_params,
        "gates": r.two_qubit_gates,
        "depth": r.transpiled_depth if r.transpiled_depth is not None else r.logical_depth,
        "evals": r.n_function_evals,
        "total_shots": r.total_shots,
        "wall_time_s": r.wall_time_s,
        "timestamp": r.timestamp,
        "error_message": r.error_message,
    }


def ranked_row(x: RankedRun) -> dict[str, Any]:
    return {
        **run_row(x.result),
        "rank": x.rank,
        "score": x.score,
        "cost": x.cost,
        "pareto_shots": x.on_front_shots,
        "pareto_gates": x.on_front_gates,
        "label": x.label,
    }


@lru_cache(maxsize=32)
def reference_energies(molecule_yaml: str, bond_length: float) -> dict[str, float]:
    mol = MoleculeConfig(**yaml.safe_load(molecule_yaml))
    ref = compute_reference(mol, bond_length)
    return {
        "e_hf": ref.e_hf,
        "e_exact": ref.e_exact,
        "hf_error_mha": abs(ref.e_hf - ref.e_exact) * 1000,
        "exact_method": ref.exact_method,
        "n_qubits_jw": ref.n_qubits_expected,
    }


@dataclass
class _Cache:
    checked: float = 0.0
    stamp: float = -1.0
    results: list[RunResult] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    derived: dict[str, Any] = field(default_factory=dict)


class Workspace:
    """The folders the app reads from, plus the configs it can launch."""

    def __init__(self, local_dir: Path, published_dir: Path, configs_dir: Path) -> None:
        self.local_dir = Path(local_dir)
        self.published_dir = Path(published_dir)
        self.configs_dir = Path(configs_dir)
        self._cache: dict[str, _Cache] = {"local": _Cache(), "published": _Cache()}
        # Changes made by this app call invalidate(); runs from the CLI are noticed this often.
        self.recheck_s = 30.0
        # The warm-up thread and the first requests can arrive together; loading must be atomic or a
        # second caller would see the cache while it is still empty (and cache derived emptiness).
        self._lock = threading.RLock()

    # ---- sources ----

    def path(self, source: str) -> Path:
        return {"local": self.local_dir, "published": self.published_dir}[source]

    def _stamp(self, source: str) -> float:
        root = self.path(source)
        newest = 0.0
        if not root.is_dir():
            return newest
        for sub in os.scandir(root):
            if not sub.is_dir() or sub.name in NON_RUN_DIRS:
                continue
            for f in os.scandir(sub.path):
                if f.name.endswith(".json"):
                    newest = max(newest, f.stat().st_mtime)
        csv = root / "summary" / RUNS_CSV
        if csv.is_file():
            newest = max(newest, csv.stat().st_mtime)
        return newest

    def _entry(self, source: str) -> _Cache:
        with self._lock:
            return self._refresh(source)

    def _refresh(self, source: str) -> _Cache:
        entry = self._cache[source]
        now = time.monotonic()
        if now - entry.checked < self.recheck_s:  # re-scanning thousands of files is slow
            return entry
        entry.checked = now
        stamp = self._stamp(source)
        if stamp != entry.stamp:
            entry.stamp = stamp
            try:
                entry.results = load_all_results(self.path(source))
            except Exception:  # noqa: BLE001 - one unreadable file must not take the app down
                entry.results = []
            entry.frame = None
            entry.derived = {}
        return entry

    def invalidate(self, source: str = "local") -> None:
        """Force a re-read on the next request (called when a job saves new results)."""
        self._cache[source].checked = 0.0

    def results(self, source: str) -> list[RunResult]:
        return self._entry(source).results

    def has_runs(self, source: str) -> bool:
        return bool(self.results(source))

    def default_source(self) -> str:
        return "local" if self.has_runs("local") else "published"

    def sources(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s,
                "label": SOURCE_LABELS[s],
                "path": str(self.path(s)),
                "n_runs": len(self.results(s)),
            }
            for s in ("published", "local")
        ]

    # ---- derived data ----

    def _memo(self, source: str, key: str, fn):
        """Compute once per change of the files; reuse the result until they change."""
        with self._lock:
            entry = self._refresh(source)
            if key not in entry.derived:
                entry.derived[key] = fn()
            return entry.derived[key]

    def ok_hashes(self, source: str) -> set[str]:
        return self._memo(
            source,
            "hashes",
            lambda: {r.config_hash for r in self.results(source) if r.status == "ok"},
        )

    def replication_runs(self, source: str) -> pd.DataFrame:
        """Per-seed rows. Published data ships them as a CSV (the JSON keeps only one seed)."""
        entry = self._entry(source)
        if entry.frame is None:
            csv = self.path(source) / "summary" / RUNS_CSV
            entry.frame = (
                normalize_runs(read_runs_csv(csv)) if csv.is_file() else runs_frame(entry.results)
            )
        return entry.frame

    def studies(self, source: str) -> list[dict[str, Any]]:
        return self._memo(
            source,
            "studies",
            lambda: clean(build_studies(self.replication_runs(source), self.hf_errors(source))),
        )

    def devices(self, source: str) -> list[dict[str, Any]]:
        return self._memo(
            source, "devices", lambda: clean(build_devices(self.replication_runs(source)))
        )

    def hf_errors(self, source: str) -> dict[str, float]:
        return self._memo(source, "hf", lambda: self._hf_errors(source))

    def _hf_errors(self, source: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.results(source):
            if r.status == "ok" and r.e_hf is not None and r.e_exact is not None:
                out.setdefault(r.config.molecule.name, abs(r.e_hf - r.e_exact) * 1000)
        return out or hf_errors(self.published_dir)

    def hardware(self, source: str) -> list[dict[str, Any]]:
        return self._memo(source, "hardware", lambda: self._hardware(source))

    def _hardware(self, source: str) -> list[dict[str, Any]]:
        labels = {r.config_hash: describe(r) for r in self.results(source) if r.status == "ok"}
        root = self.path(source)
        hw_dir = root / "hardware"
        if not hw_dir.is_dir() and source == "local":
            hw_dir = root.parent / "hardware" if (root.parent / "hardware").is_dir() else hw_dir
        return clean(build_hardware(hw_dir, labels)) if hw_dir.is_dir() else []

    def curves(self, source: str) -> list[dict[str, Any]]:
        d = self.path(source) / "curves"
        return self._memo(source, "curves", lambda: clean(build_curves(d)) if d.is_dir() else [])

    def hardware_available(self, source: str) -> int:
        return len(self.hardware(source))

    def find_run(self, source: str, run_hash: str) -> RunResult | None:
        for sub in self.path(source).glob("*"):
            f = sub / f"{run_hash}.json"
            if sub.is_dir() and sub.name not in NON_RUN_DIRS and f.is_file():
                return load_result(f)
        return None

    # ---- configs ----

    def molecules(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted((self.configs_dir / "molecules").glob("*.yaml")):
            data = yaml.safe_load(p.read_text())
            key = p.stem.lower()
            eq = EQUILIBRIUM.get(key, 1.0)
            out.append(
                {
                    "id": key,
                    "name": data["name"],
                    "geometry": data["geometry"],
                    "basis": data.get("basis", "sto3g"),
                    "active_electrons": data.get("active_electrons"),
                    "active_orbitals": data.get("active_orbitals"),
                    "equilibrium": eq,
                    "bond_lengths": default_bond_lengths(key),
                }
            )
        return out

    def molecule_yaml(self, molecule_id: str) -> str:
        p = self.configs_dir / "molecules" / f"{molecule_id.lower()}.yaml"
        if not p.is_file():
            raise FileNotFoundError(molecule_id)
        return p.read_text()

    def sweep_path(self, name: str) -> Path:
        p = self.configs_dir / "sweeps" / f"{Path(name).stem}.yaml"
        if not p.is_file():
            raise FileNotFoundError(name)
        return p
