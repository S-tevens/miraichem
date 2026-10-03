"""Fixed-parameter hardware evaluation of a saved VQE run.

Take the optimal parameters from a saved simulator run, evaluate the energy on a real device once
per mitigation setting, and store it next to the simulator's prediction and the exact energy. The
point: does the noisy simulator predict what the real chip does?
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from miraichem.ansatz import build_ansatz
from miraichem.backends.hardware import (
    DEFAULT_HARDWARE_DIR,
    HardwareAbortedError,
    HardwareBackend,
    HardwarePlan,
    estimate_plan,
    get_service,
    remaining_qpu_seconds,
    resolve_backend,
)
from miraichem.backends.ideal import IdealBackend
from miraichem.benchmark.metrics import abs_error_mha
from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR, load_result
from miraichem.config import MitigationKind
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.vqe.result import RunResult
from miraichem.vqe.runner import library_versions


class HardwareEvaluation(BaseModel):
    mitigation: str
    e_hw: float
    e_hw_std: float
    abs_error_mha: float
    job_id: str


class HardwareResult(BaseModel):
    """One hardware comparison: simulator prediction vs real device, per mitigation mode."""

    source_config_hash: str
    molecule: str
    bond_length: float
    backend: str
    shots: int
    parameters: list[float]
    e_exact: float
    e_hf: float
    e_ideal_at_params: float  # noiseless energy of this circuit at these parameters
    e_simulator: float  # what the noisy simulator reported for the source run
    simulator_mitigation: str
    evaluations: list[HardwareEvaluation]
    timestamp: str
    library_versions: dict[str, str] = {}


def load_source_run(config_hash: str, molecule: str, results_dir: Path) -> RunResult:
    path = Path(results_dir) / molecule.lower() / f"{config_hash}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved run {config_hash} for {molecule} in {results_dir}.")
    run = load_result(path)
    if run.status != "ok" or not run.optimal_parameters:
        raise ValueError(f"Run {config_hash} did not succeed; cannot evaluate it on hardware.")
    return run


def plan_evaluation(
    source: RunResult,
    modes: list[MitigationKind],
    shots: int,
    backend_name: str | None = None,
    service: Any = None,
) -> HardwarePlan:
    """Cost plan for evaluating ``source`` on hardware. Read-only: submits nothing.

    Uses the account (if a token is set) only to name the backend and read the remaining budget.
    """
    cfg = source.config
    problem = build_qubit_hamiltonian(cfg.molecule, cfg.bond_length, cfg.mapping)
    groups = len(problem.hamiltonian.group_commuting(qubit_wise=True))
    name, remaining = backend_name or "least busy (chosen at submission)", None
    try:
        service = service or get_service()
        name = resolve_backend(service, backend_name, problem.num_qubits).name
        remaining = remaining_qpu_seconds(service)
    except Exception:  # noqa: BLE001 - dry run must work offline / without a token
        pass
    return estimate_plan(name, modes, shots, groups, remaining)


def run_hardware_evaluation(
    source: RunResult,
    modes: list[MitigationKind],
    shots: int,
    confirm: Callable[[HardwarePlan], bool],
    backend_name: str | None = None,
    service: Any = None,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    estimator_factory: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> HardwareResult:
    """Evaluate the source run's optimal parameters on hardware (ONE combined confirmation).

    Raises HardwareAbortedError (nothing submitted) if the plan is declined.
    """
    assert source.e_exact is not None and source.e_hf is not None and source.e_vqe is not None
    cfg = source.config
    problem = build_qubit_hamiltonian(cfg.molecule, cfg.bond_length, cfg.mapping)
    circuit = build_ansatz(cfg.ansatz, problem, cfg.ansatz_reps).circuit
    theta = np.asarray(source.optimal_parameters, dtype=float)
    service = service or get_service()

    plan = plan_evaluation(source, modes, shots, backend_name, service)
    if not confirm(plan):
        raise HardwareAbortedError("Hardware submission was not confirmed; nothing was sent.")

    e_ideal = IdealBackend().estimate(circuit, problem.hamiltonian, theta).value
    e_ideal += problem.energy_shift
    evaluations, backend_used = [], plan.backend_name
    for mode in modes:
        kwargs: dict[str, Any] = {}
        if estimator_factory is not None:
            kwargs["estimator_factory"] = estimator_factory
        hw = HardwareBackend(
            allow_hardware=True,
            confirm=lambda _plan: True,  # already confirmed once for all jobs above
            service=service,
            backend_name=backend_name or plan.backend_name,
            shots=shots,
            mitigation=mode,
            hardware_dir=hardware_dir,
            seed=cfg.seed,
            context={
                "source_config_hash": source.config_hash,
                "molecule": cfg.molecule.name,
                "energy_shift": problem.energy_shift,
            },
            **kwargs,
        )
        est = hw.estimate(circuit, problem.hamiltonian, theta)
        e_hw = est.value + problem.energy_shift
        backend_used = est.metadata["backend"]
        evaluations.append(
            HardwareEvaluation(
                mitigation=mode,
                e_hw=e_hw,
                e_hw_std=est.std_error,
                abs_error_mha=abs_error_mha(e_hw, source.e_exact),
                job_id=est.metadata["job_id"],
            )
        )
    return HardwareResult(
        source_config_hash=source.config_hash,
        molecule=cfg.molecule.name,
        bond_length=cfg.bond_length,
        backend=backend_used,
        shots=shots,
        parameters=[float(x) for x in theta],
        e_exact=source.e_exact,
        e_hf=source.e_hf,
        e_ideal_at_params=e_ideal,
        e_simulator=source.e_vqe,
        simulator_mitigation=cfg.mitigation,
        evaluations=evaluations,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        library_versions=library_versions(),
    )


def save_hardware_result(result: HardwareResult, hardware_dir: Path = DEFAULT_HARDWARE_DIR) -> Path:
    path = Path(hardware_dir) / result.molecule.lower() / f"{result.source_config_hash}_hw.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))
    return path


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "HardwareEvaluation",
    "HardwareResult",
    "load_source_run",
    "plan_evaluation",
    "run_hardware_evaluation",
    "save_hardware_result",
]
