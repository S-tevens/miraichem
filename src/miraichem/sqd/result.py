"""The record saved for every SQD run (separate from the VQE ``RunResult``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from miraichem.sqd.config import SqdConfig


class SqdResult(BaseModel):
    """Inputs, outputs and provenance of one SQD run. Energies in Hartree."""

    config: SqdConfig
    config_hash: str
    status: Literal["ok", "failed"] = "ok"
    error_message: str | None = None

    e_sqd: float | None = None
    e_hf: float | None = None
    e_exact: float | None = None  # FCI in the active space
    abs_error_mha: float | None = None
    hf_error_mha: float | None = None
    within_chemical_accuracy: bool | None = None

    num_qubits: int | None = None
    n_determinants: int | None = None  # size of the full CI space
    n_distinct_samples: int | None = None
    n_postselected: int | None = None  # samples with the right electron count after the last round
    subspace_dim: int | None = None  # dimension of the diagonalised subspace (best batch)
    subspace_fraction: float | None = None  # subspace_dim relative to the full CI space
    error_per_iteration_mha: list[float] = []
    wall_time_s: float | None = None

    library_versions: dict[str, str] = {}
    git_commit: str | None = None
    timestamp: str | None = None
