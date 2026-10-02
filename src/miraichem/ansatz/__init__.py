"""Ansatz circuits (parameterised guesses for the ground state)."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from miraichem.ansatz.hardware_efficient import build_hea
from miraichem.ansatz.uccsd import build_uccsd, hartree_fock_circuit
from miraichem.config import AnsatzKind
from miraichem.mapping.hamiltonian import QubitProblem


@dataclass
class AnsatzInfo:
    """An ansatz circuit with its logical (pre-transpilation) metadata."""

    circuit: QuantumCircuit
    num_parameters: int
    logical_depth: int  # depth after unrolling to elementary gates, before hardware transpilation


def build_ansatz(kind: AnsatzKind, problem: QubitProblem, reps: int = 1) -> AnsatzInfo:
    """Build the requested ansatz, starting from the Hartree-Fock state for the mapping."""
    if kind == "uccsd":
        circuit = build_uccsd(problem, reps)
    elif kind == "hea":
        circuit = build_hea(problem, reps)
    else:
        raise ValueError(f"Unknown ansatz: {kind!r}")
    unrolled = circuit.decompose(reps=8)
    return AnsatzInfo(circuit, circuit.num_parameters, unrolled.depth())


__all__ = ["AnsatzInfo", "build_ansatz", "build_hea", "build_uccsd", "hartree_fock_circuit"]
