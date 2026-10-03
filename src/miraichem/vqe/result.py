"""The record saved for every VQE run."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from miraichem.config import RunConfig


class RunResult(BaseModel):
    """Everything about one run: inputs, outputs, cost and provenance. Energies in Hartree."""

    config: RunConfig
    config_hash: str
    status: Literal["ok", "failed"] = "ok"
    error_message: str | None = None

    e_vqe: float | None = None
    e_vqe_unmitigated: float | None = None
    e_hf: float | None = None
    e_exact: float | None = None
    abs_error_mha: float | None = None
    within_chemical_accuracy: bool | None = None

    num_qubits: int | None = None
    num_params: int | None = None
    logical_depth: int | None = None
    transpiled_depth: int | None = None
    two_qubit_gates: int | None = None
    n_function_evals: int | None = None
    total_shots: int | None = None
    wall_time_s: float | None = None

    convergence_trace: list[float] = []  # total energy of every evaluation, in order
    convergence_trace_unmitigated: list[float] = []  # raw energies, when mitigation is on
    optimal_parameters: list[float] = []
    backend_info: dict[str, Any] = {}
    job_ids: list[str] = []
    library_versions: dict[str, str] = {}
    git_commit: str | None = None
    timestamp: str | None = None
