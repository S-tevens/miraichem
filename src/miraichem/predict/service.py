"""Glue between the predictor and the rest of the package: pre-run circuit metrics for a config."""

from __future__ import annotations

from functools import lru_cache

from miraichem.ansatz import build_ansatz
from miraichem.backends.noisy import NoisyBackend
from miraichem.chemistry.classical import compute_reference
from miraichem.config import RunConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.predict.features import RunShape


@lru_cache(maxsize=256)
def _shape(config_json: str) -> RunShape:
    cfg = RunConfig.model_validate_json(config_json)
    problem = build_qubit_hamiltonian(cfg.molecule, cfg.bond_length, cfg.mapping)
    ansatz = build_ansatz(cfg.ansatz, problem, cfg.ansatz_reps)
    metrics = NoisyBackend(cfg.fake_backend_name, cfg.shots, cfg.seed).transpiled_metrics(
        ansatz.circuit, problem.hamiltonian
    )
    ref = compute_reference(cfg.molecule, cfg.bond_length)
    return RunShape(
        config=cfg,
        two_qubit_gates=float(metrics["two_qubit_gates"]),
        depth=float(metrics["transpiled_depth"]),
        qubits=problem.num_qubits,
        parameters=ansatz.num_parameters,
        hf_error_mha=abs(ref.e_hf - ref.e_exact) * 1000,
    )


def shape_for_config(cfg: RunConfig) -> RunShape:
    """Transpile the circuit for the device (no optimisation) to learn its size before running.

    Only the molecule, geometry, mapping, circuit and device shape the transpiled circuit. The
    optimizer, budget, mitigation, shots and seed do not, so they are neutralised in the cache key
    and changing them gives an instant answer.
    """
    key = cfg.model_copy(
        update={
            "optimizer": "cobyla",
            "maxiter": 1,
            "shots": 1024,
            "mitigation": "none",
            "zne_scales": [1.0, 3.0],
            "zne_method": "linear",
            "seed": 0,
        }
    )
    shape = _shape(key.model_dump_json())
    return RunShape(**{**shape.__dict__, "config": cfg})
