"""Readout (measurement) error mitigation.

At the end of a circuit each qubit is measured, and the measurement itself is imperfect: a qubit
that is really |0> is sometimes read as 1 and vice versa. We measure how often this happens by
preparing known states (all |0>, all |1>), build each qubit's 2x2 "confusion" matrix, and invert it
to correct the measured outcome distribution. This is the cheapest mitigation (2 extra circuits).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np


@dataclass(frozen=True)
class ReadoutCalibration:
    """Per-qubit readout error rates, indexed by physical qubit."""

    p1_given_0: dict[int, float]  # P(read 1 | prepared 0)
    p0_given_1: dict[int, float]  # P(read 0 | prepared 1)

    def assignment_matrix(self, qubit: int) -> np.ndarray:
        """Columns = true state (0, 1); rows = measured state (0, 1)."""
        e0, e1 = self.p1_given_0[qubit], self.p0_given_1[qubit]
        return np.array([[1 - e0, e1], [e0, 1 - e1]])

    def correct(self, probs: np.ndarray, qubits: list[int]) -> np.ndarray:
        """Undo readout error on a distribution over ``qubits``.

        ``probs[i]`` is the probability of the outcome whose bit j equals the result of qubits[j]
        (bit 0 is the least significant). The result can contain tiny negative values from
        statistical noise; we leave them in so the corrected expectation values stay unbiased.
        """
        # kron order: most significant bit first.
        inv = [np.linalg.inv(self.assignment_matrix(q)) for q in reversed(qubits)]
        return reduce(np.kron, inv) @ probs


def calibration_from_counts(
    counts_zero: dict[str, int], counts_one: dict[str, int], qubits: list[int]
) -> ReadoutCalibration:
    """Estimate error rates from the counts of the all-|0> and all-|1> calibration circuits.

    Bit j of a count string (read right to left) is the result for ``qubits[j]``.
    """
    p10: dict[int, float] = {}
    p01: dict[int, float] = {}
    shots_zero, shots_one = sum(counts_zero.values()), sum(counts_one.values())
    for j, q in enumerate(qubits):
        ones_when_0 = sum(c for k, c in counts_zero.items() if k[::-1][j] == "1")
        zeros_when_1 = sum(c for k, c in counts_one.items() if k[::-1][j] == "0")
        p10[q] = ones_when_0 / shots_zero
        p01[q] = zeros_when_1 / shots_one
    return ReadoutCalibration(p1_given_0=p10, p0_given_1=p01)
