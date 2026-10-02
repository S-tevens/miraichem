"""Ideal (noise-free, infinite-shot) backend using exact statevector simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from miraichem.backends.base import EstimateResult


class IdealBackend:
    """Exact expectation values from the statevector. The best case: no noise, no shot noise."""

    def __init__(self) -> None:
        self._estimator = StatevectorEstimator()

    def estimate(
        self, circuit: QuantumCircuit, observable: SparsePauliOp, parameter_values: np.ndarray
    ) -> EstimateResult:
        pub = (circuit, observable, np.asarray(parameter_values, dtype=float))
        value = float(self._estimator.run([pub]).result()[0].data.evs)
        return EstimateResult(value=value, std_error=0.0)

    def info(self) -> dict[str, Any]:
        return {"name": "statevector", "kind": "ideal"}

    @property
    def total_shots(self) -> int:
        return 0
