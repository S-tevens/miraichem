"""Zero-noise extrapolation (ZNE).

Idea: we cannot turn hardware noise off, but we CAN turn it up. Run the same circuit at several
amplified noise levels, see how the energy drifts as noise grows, and extrapolate the trend back
to "zero noise". Noise is amplified here by gate folding: every two-qubit gate G is replaced by
G G G (G is its own inverse, so the ideal result is unchanged but the gate runs 3 times and
accumulates 3x the error). Two-qubit gates dominate the error on IBM chips, so we scale only them
and a "scale factor" below means the multiple of the two-qubit-gate noise.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

# Two-qubit gates native to IBM devices; all are self-inverse, so G G G == G ideally.
SELF_INVERSE_2Q = {"ecr", "cz", "cx"}

Method = str  # "linear" | "richardson"


def fold_two_qubit_gates(circuit: QuantumCircuit, scale: float) -> QuantumCircuit:
    """Return a copy with two-qubit noise amplified by ``scale`` (>= 1) via gate folding.

    scale 1 leaves the circuit alone, 3 repeats every two-qubit gate 3 times, and values in
    between fold only a leading fraction of the gates (partial folding).
    """
    if scale < 1:
        raise ValueError("ZNE scale factors must be >= 1.")
    gate_idx = [
        i
        for i, inst in enumerate(circuit.data)
        if len(inst.qubits) == 2 and inst.operation.name not in {"barrier", "delay"}
    ]
    bad = {circuit.data[i].operation.name for i in gate_idx} - SELF_INVERSE_2Q
    if bad:
        raise ValueError(f"Cannot fold two-qubit gates {sorted(bad)}; only {SELF_INVERSE_2Q}.")
    extra_pairs = (scale - 1.0) / 2.0 * len(gate_idx)  # number of extra G-G pairs to insert
    base, remainder = divmod(extra_pairs, len(gate_idx)) if gate_idx else (0, 0)
    base, remainder = int(base), int(round(remainder))

    folded = circuit.copy_empty_like()
    position = {i: n for n, i in enumerate(gate_idx)}
    for i, inst in enumerate(circuit.data):
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if i in position:
            repeats = base + (1 if position[i] < remainder else 0)
            for _ in range(repeats):
                folded.append(inst.operation, inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded


def extrapolation_weights(scales: list[float], method: Method) -> np.ndarray:
    """Weights w with value_at_zero_noise = sum(w_i * E_i) for energies E_i at the given scales.

    ``linear`` is a least-squares straight line; ``richardson`` is the exact polynomial through all
    points. Both are linear in the data, so the statistical error also follows from the weights.
    """
    x = np.asarray(scales, dtype=float)
    if len(x) < 2:
        raise ValueError("ZNE needs at least two scale factors.")
    if method == "linear":
        degree = 1
    elif method == "richardson":
        degree = len(x) - 1
    else:
        raise ValueError(f"Unknown extrapolation method {method!r}.")
    vander = np.vander(x, degree + 1, increasing=True)  # columns: 1, x, x^2, ...
    return (np.linalg.pinv(vander))[0]  # intercept row


def extrapolate(
    scales: list[float], values: list[float], variances: list[float], method: Method = "linear"
) -> tuple[float, float]:
    """Extrapolate to zero noise. Returns (value, standard_error)."""
    w = extrapolation_weights(scales, method)
    value = float(w @ np.asarray(values))
    std = float(np.sqrt(np.sum(w**2 * np.asarray(variances))))
    return value, std
