"""Common interface for energy estimators (ideal, noisy simulator, hardware)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


@dataclass
class EstimateResult:
    """Expectation value of an observable. Hamiltonian units: Hartree (without the energy shift)."""

    value: float
    std_error: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EnergyBackend(Protocol):
    """Anything that can estimate <psi(theta)|H|psi(theta)>."""

    def estimate(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        parameter_values: np.ndarray,
    ) -> EstimateResult: ...

    def info(self) -> dict[str, Any]:
        """Static description of the backend (name, shots, transpiled depth, ...)."""
        ...

    @property
    def total_shots(self) -> int:
        """Shots used so far (0 for exact simulation)."""
        ...
