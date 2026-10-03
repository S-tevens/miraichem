"""Real IBM quantum hardware via Qiskit Runtime (EstimatorV2), behind several safety gates.

Hardware time is scarce (free plan: about 600 s of QPU time per period), so by default we do NOT
run a whole VQE optimisation on a device. Instead we take the optimal parameters found on the noisy
simulator and evaluate the energy ONCE per mitigation setting on hardware ("fixed-parameter
evaluation"). That costs a handful of short jobs instead of hundreds, and it directly answers
"does the simulator's prediction match a real chip?".

Safety gates, all enforced in code:
  1. ``allow_hardware`` must be True (the CLI flag ``--allow-hardware``), else nothing is created.
  2. A cost plan (circuits, shots, estimated QPU seconds) is printed and must be confirmed.
  3. Every submitted job ID is written to disk immediately, so results can be re-fetched if the
     session dies. The token comes from ``.env`` (never hard-coded).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from miraichem.backends.base import EstimateResult
from miraichem.config import MitigationKind

DEFAULT_HARDWARE_DIR = Path("results") / "hardware"
FREE_PLAN_BUDGET_S = 600.0  # free-plan QPU seconds per usage period (checked 2026-10-02)
PER_JOB_OVERHEAD_S = 3.0  # fixed cost per job (loading, readout setup), conservative
PER_SHOT_S = 0.0004  # ~250 us repetition delay + circuit duration, per shot per circuit
SAFETY_FACTOR = 1.5
ZNE_NOISE_FACTORS = (1, 3, 5)


class HardwareDisabledError(RuntimeError):
    """Raised when hardware is requested without the explicit ``--allow-hardware`` opt-in."""


class HardwareAbortedError(RuntimeError):
    """Raised when the user declines the cost confirmation (nothing was submitted)."""


@dataclass
class HardwarePlan:
    """What we are about to run, and what it should cost. Printed before any submission."""

    backend_name: str
    modes: list[str]
    shots: int
    n_jobs: int
    n_circuits: int  # total circuits across all jobs
    estimated_qpu_seconds: float
    budget_seconds: float = FREE_PLAN_BUDGET_S
    remaining_seconds: float | None = None

    def lines(self) -> list[str]:
        out = [
            f"Backend         : {self.backend_name}",
            f"Mitigation modes: {', '.join(self.modes)}",
            f"Jobs            : {self.n_jobs}",
            f"Circuits (total): {self.n_circuits}",
            f"Shots per circuit: {self.shots}",
            f"Estimated QPU time: ~{self.estimated_qpu_seconds:.0f} s (rough upper-bound estimate)",
        ]
        if self.remaining_seconds is not None:
            out.append(
                f"QPU time remaining this period: {self.remaining_seconds:.0f} s "
                f"of {self.budget_seconds:.0f} s"
            )
        return out


def mitigation_options(mitigation: MitigationKind, shots: int) -> dict[str, Any]:
    """Runtime EstimatorV2 options for one mitigation setting.

    Runtime's built-in resilience is used on hardware (instead of our simulator-side code):
    level 0 = none; level 1 = readout (TREX) mitigation; level 2 = ZNE (+ gate twirling).
    ``zne`` alone switches the readout part of level 2 off so the two can be compared.
    """
    base: dict[str, Any] = {"default_shots": shots}
    if mitigation == "none":
        return {**base, "resilience_level": 0}
    if mitigation == "readout":
        return {**base, "resilience_level": 1}
    zne = {"noise_factors": list(ZNE_NOISE_FACTORS)}
    if mitigation == "readout+zne":
        return {**base, "resilience_level": 2, "resilience": {"zne": zne}}
    if mitigation == "zne":
        return {
            **base,
            "resilience_level": 2,
            "resilience": {"measure_mitigation": False, "zne": zne},
        }
    raise ValueError(f"Unknown mitigation {mitigation!r}.")


def circuits_per_job(num_groups: int, mitigation: MitigationKind) -> int:
    """Rough number of circuits one evaluation turns into (measurement groups x ZNE factors)."""
    multiplicity = len(ZNE_NOISE_FACTORS) if "zne" in mitigation else 1
    return num_groups * multiplicity


def estimate_plan(
    backend_name: str,
    modes: list[MitigationKind],
    shots: int,
    num_groups: int,
    remaining_seconds: float | None = None,
) -> HardwarePlan:
    """Cost plan for one evaluation per mode. A deliberately conservative upper-bound estimate."""
    total_circuits, seconds = 0, 0.0
    for mode in modes:
        n = circuits_per_job(num_groups, mode)
        readout_factor = 2.0 if mode in ("readout", "readout+zne") else 1.0  # TREX extra shots
        seconds += PER_JOB_OVERHEAD_S + n * shots * PER_SHOT_S * readout_factor
        total_circuits += n
    return HardwarePlan(
        backend_name=backend_name,
        modes=[str(m) for m in modes],
        shots=shots,
        n_jobs=len(modes),
        n_circuits=total_circuits,
        estimated_qpu_seconds=seconds * SAFETY_FACTOR,
        remaining_seconds=remaining_seconds,
    )


def get_service():
    """QiskitRuntimeService from IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE (loaded from .env)."""
    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    return QiskitRuntimeService(
        channel="ibm_cloud", token=token, instance=os.getenv("IBM_QUANTUM_INSTANCE") or None
    )


def remaining_qpu_seconds(service) -> float | None:
    """Remaining QPU seconds this period, or None if the usage API is unavailable."""
    try:
        return float(service.usage()["usage_remaining_seconds"])
    except Exception:  # noqa: BLE001 - usage is informational only
        return None


def resolve_backend(service, name: str | None, min_qubits: int):
    """Named backend, or the least-busy operational real device with enough qubits."""
    if name:
        return service.backend(name)
    return service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)


def _default_estimator_factory(backend, options: dict[str, Any]):
    from qiskit_ibm_runtime import EstimatorV2

    return EstimatorV2(mode=backend, options=options)  # job mode (the only mode on the free plan)


def record_job(hardware_dir: Path, job_id: str, info: dict[str, Any]) -> Path:
    """Write the job ID (and context needed to interpret it) to disk right after submission."""
    path = Path(hardware_dir) / "jobs" / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"job_id": job_id, **info}, indent=2))
    return path


@dataclass
class HardwareBackend:
    """Energy estimator that submits one Runtime job per ``estimate`` call.

    Args:
        allow_hardware: must be True; otherwise ``HardwareDisabledError`` is raised on creation.
        confirm: called with the cost plan before submitting; must return True to proceed.
        service: a QiskitRuntimeService (created from .env when omitted).
        backend_name: a specific device, or None for the least busy one.
        mitigation: Runtime resilience setting for this backend's evaluations.
        estimator_factory: builds the Runtime estimator (replaceable in tests).
    """

    allow_hardware: bool
    confirm: Callable[[HardwarePlan], bool] | None = None
    service: Any = None
    backend_name: str | None = None
    shots: int = 4096
    mitigation: MitigationKind = "none"
    hardware_dir: Path = DEFAULT_HARDWARE_DIR
    optimization_level: int = 3
    seed: int = 42
    estimator_factory: Callable[[Any, dict[str, Any]], Any] = _default_estimator_factory
    context: dict[str, Any] = field(default_factory=dict)  # extra info stored with the job record
    _backend: Any = field(default=None, init=False, repr=False)
    _job_ids: list[str] = field(default_factory=list, init=False)
    _shots_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.allow_hardware:
            raise HardwareDisabledError(
                "Real hardware is disabled. Pass --allow-hardware to enable it."
            )

    def _device(self, min_qubits: int):
        if self.service is None:
            self.service = get_service()
        if self._backend is None:
            self._backend = resolve_backend(self.service, self.backend_name, min_qubits)
        return self._backend

    def plan(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> HardwarePlan:
        device = self._device(circuit.num_qubits)
        groups = len(observable.group_commuting(qubit_wise=True))
        return estimate_plan(
            device.name, [self.mitigation], self.shots, groups, remaining_qpu_seconds(self.service)
        )

    def estimate(
        self, circuit: QuantumCircuit, observable: SparsePauliOp, parameter_values: np.ndarray
    ) -> EstimateResult:
        device = self._device(circuit.num_qubits)
        plan = self.plan(circuit, observable)
        if self.confirm is None or not self.confirm(plan):
            raise HardwareAbortedError("Hardware submission was not confirmed; nothing was sent.")

        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=device, seed_transpiler=self.seed
        )
        isa = pm.run(circuit)
        isa_obs = observable.apply_layout(isa.layout)
        estimator = self.estimator_factory(device, mitigation_options(self.mitigation, self.shots))
        job = estimator.run([(isa, isa_obs, np.asarray(parameter_values, dtype=float))])
        job_id = str(job.job_id())
        self._job_ids.append(job_id)
        record_job(
            self.hardware_dir,
            job_id,
            {
                "backend": device.name,
                "mitigation": self.mitigation,
                "shots": self.shots,
                "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
                **self.context,
            },
        )
        data = job.result()[0].data
        self._shots_used += self.shots * circuits_per_job(
            len(isa_obs.group_commuting(qubit_wise=True)), self.mitigation
        )
        return EstimateResult(
            value=float(data.evs),
            std_error=float(np.max(data.stds)) if hasattr(data, "stds") else 0.0,
            metadata={
                "job_id": job_id,
                "backend": device.name,
                "mitigation": self.mitigation,
                "transpiled_depth": isa.depth(),
            },
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.backend_name or "least_busy",
            "kind": "hardware",
            "backend": getattr(self._backend, "name", self.backend_name),
            "shots": self.shots,
            "mitigation": self.mitigation,
            "job_ids": list(self._job_ids),
        }

    @property
    def total_shots(self) -> int:
        return self._shots_used

    @property
    def job_ids(self) -> list[str]:
        return list(self._job_ids)


def fetch_job(service, job_id: str) -> dict[str, Any]:
    """Re-fetch a finished job's expectation value (works after a crashed session)."""
    job = service.job(job_id)
    data = job.result()[0].data
    return {
        "job_id": job_id,
        "status": str(job.status()),
        "value": float(data.evs),
        "std_error": float(np.max(data.stds)) if hasattr(data, "stds") else None,
    }
