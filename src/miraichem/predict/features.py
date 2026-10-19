"""Features for predicting a run's error before running it.

Every feature is something known BEFORE the optimisation: the circuit (after transpilation for the
device), the optimizer and its budget, the mitigation, the shot count, the molecule's Hartree-Fock
error and a summary of the device's noise. The target is log10 of the final error in mHa, because
errors span four orders of magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult

FEATURES = [
    "log_two_qubit_gates",
    "log_depth",
    "qubits",
    "parameters",
    "log2_shots",
    "log_budget",
    "readout",
    "zne",
    "opt_cobyla",
    "opt_spsa",
    "opt_spsa_clip",
    "ansatz_uccsd",
    "ansatz_hea",
    "ansatz_hea_hf",
    "layers",
    "parity",
    "log_hf_error",
    "log_device_gate_error",
    "log_device_readout_error",
]

FEATURE_LABELS = {
    "log_two_qubit_gates": "two-qubit gate count",
    "log_depth": "circuit depth",
    "qubits": "qubits",
    "parameters": "circuit parameters",
    "log2_shots": "shots per circuit",
    "log_budget": "evaluation budget",
    "readout": "readout correction",
    "zne": "zero-noise extrapolation",
    "opt_cobyla": "optimizer: COBYLA",
    "opt_spsa": "optimizer: SPSA",
    "opt_spsa_clip": "optimizer: clipped SPSA",
    "ansatz_uccsd": "circuit: UCCSD",
    "ansatz_hea": "circuit: hardware-efficient",
    "ansatz_hea_hf": "circuit: hardware-efficient, HF start",
    "layers": "circuit layers",
    "parity": "Parity mapping",
    "log_hf_error": "molecule's Hartree-Fock error",
    "log_device_gate_error": "device two-qubit gate error",
    "log_device_readout_error": "device readout error",
}


@lru_cache(maxsize=32)
def device_noise(name: str) -> tuple[float, float]:
    """Median two-qubit gate error and median readout error of a fake IBM device.

    Medians, not means: a few very bad qubit pairs would otherwise dominate the summary.
    """
    from qiskit_ibm_runtime import fake_provider

    target = getattr(fake_provider, name)().target
    gate_errors: list[float] = []
    for op in ("ecr", "cz", "cx"):
        if op in target.operation_names:
            gate_errors = [
                p.error for p in target[op].values() if p is not None and p.error is not None
            ]
            break
    readout = [p.error for p in target["measure"].values() if p is not None and p.error is not None]
    return float(np.median(gate_errors)), float(np.median(readout))


@dataclass(frozen=True)
class RunShape:
    """What a run looks like before it is run."""

    config: RunConfig
    two_qubit_gates: float
    depth: float
    qubits: int
    parameters: int
    hf_error_mha: float


def vector(s: RunShape) -> list[float]:
    c = s.config
    gate_err, ro_err = device_noise(c.fake_backend_name)
    return [
        math.log10(s.two_qubit_gates + 1),
        math.log10(s.depth + 1),
        float(s.qubits),
        float(s.parameters),
        math.log2(c.shots),
        math.log10(c.maxiter),
        float(c.mitigation in ("readout", "readout+zne")),
        float(c.mitigation in ("zne", "readout+zne")),
        float(c.optimizer == "cobyla"),
        float(c.optimizer == "spsa"),
        float(c.optimizer == "spsa_clip"),
        float(c.ansatz == "uccsd"),
        float(c.ansatz == "hea"),
        float(c.ansatz == "hea_hf"),
        float(c.ansatz_reps),
        float(c.mapping == "parity"),
        math.log10(max(s.hf_error_mha, 1e-3)),
        math.log10(gate_err),
        math.log10(ro_err),
    ]


def shape_of(r: RunResult) -> RunShape | None:
    """The pre-run description of a finished noisy run, or None if something is missing."""
    if (
        r.status != "ok"
        or r.config.backend != "noisy"
        or r.abs_error_mha is None
        or r.e_hf is None
        or r.e_exact is None
        or r.num_qubits is None
        or r.num_params is None
    ):
        return None
    depth = r.transpiled_depth if r.transpiled_depth is not None else r.logical_depth
    return RunShape(
        config=r.config,
        two_qubit_gates=float(r.two_qubit_gates or 0),
        depth=float(depth or 0),
        qubits=int(r.num_qubits),
        parameters=int(r.num_params),
        hf_error_mha=abs(r.e_hf - r.e_exact) * 1000,
    )


def target_of(r: RunResult) -> float:
    return math.log10(max(float(r.abs_error_mha or 0.0), 1e-3))


def group_key(c: RunConfig) -> str:
    """Runs of the same configuration (any seed) share a key, so validation never leaks."""
    return c.model_dump_json(exclude={"seed"})
