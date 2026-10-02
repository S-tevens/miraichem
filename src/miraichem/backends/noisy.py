"""Noisy simulator: Aer with the noise model, gate set and coupling map of a fake IBM backend.

Unlike a plain noiseless simulation, this samples a finite number of shots, applies gate and
readout errors copied from a real device's calibration snapshot, and runs on the transpiled
circuit (routed onto the device's connectivity). It is our stand-in for real hardware during
development, so results here predict how a configuration will behave on a real chip.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider

from miraichem.backends.base import EstimateResult

_TWO_QUBIT_FREE = {"barrier", "delay"}


def load_fake_backend(name: str):
    """Instantiate a fake IBM backend from ``qiskit_ibm_runtime.fake_provider`` by class name."""
    try:
        return getattr(fake_provider, name)()
    except AttributeError as exc:
        raise ValueError(f"Unknown fake backend {name!r}.") from exc


def _trim_noise_model(model: NoiseModel, qubits: set[int]) -> NoiseModel:
    """Keep only the errors that act entirely on ``qubits``.

    Aer re-processes the whole noise model on every call, so a 127-qubit device is ~5x slower
    than a 7-qubit one even for a 4-qubit circuit. Dropping errors on unused qubits does not change
    the physics of the circuit (idle qubits are discarded by Aer anyway) but makes runs fast.
    Uses NoiseModel's private error tables, which is fine because versions are pinned.
    """
    trimmed = NoiseModel(basis_gates=model.basis_gates)
    for inst, by_qubits in model._local_quantum_errors.items():
        for qs, err in by_qubits.items():
            if set(qs) <= qubits:
                trimmed.add_quantum_error(err, inst, list(qs))
    for qs, err in model._local_readout_errors.items():
        if set(qs) <= qubits:
            trimmed.add_readout_error(err, list(qs))
    return trimmed


class NoisyBackend:
    """Shot-based estimator on an Aer simulator that mimics a named fake IBM backend."""

    def __init__(
        self,
        fake_backend_name: str = "FakeSherbrooke",
        shots: int = 4096,
        seed: int = 42,
        optimization_level: int = 3,
    ) -> None:
        self.fake_backend_name = fake_backend_name
        self.shots = shots
        self.seed = seed
        self.optimization_level = optimization_level
        self._fake = load_fake_backend(fake_backend_name)
        fake = self._fake
        # from_backend copies the noise model, basis gates and coupling map of the device.
        self._sim = AerSimulator.from_backend(fake, seed_simulator=seed)
        self._estimator: BackendEstimatorV2 | None = None
        self._cache: tuple[int, QuantumCircuit, SparsePauliOp, int] | None = None
        self._evals = 0
        self._shots_used = 0

    def _prepare(self, circuit: QuantumCircuit, observable: SparsePauliOp):
        """Transpile once to the device (transpilation is slow; parameters are bound later)."""
        key = id(circuit)
        if self._cache is None or self._cache[0] != key:
            pm = generate_preset_pass_manager(
                optimization_level=self.optimization_level,
                backend=self._sim,
                seed_transpiler=self.seed,
            )
            isa = pm.run(circuit)
            used = {isa.find_bit(q).index for inst in isa.data for q in inst.qubits}
            # Same device target (needed by the estimator's own basis-change transpilation),
            # but only the noise that matters for the qubits this circuit uses.
            run_sim = AerSimulator.from_backend(
                self._fake,
                noise_model=_trim_noise_model(self._sim.options.noise_model, used),
                seed_simulator=self.seed,
            )
            self._estimator = BackendEstimatorV2(
                backend=run_sim,
                options={
                    "default_precision": 1.0 / math.sqrt(self.shots),
                    "seed_simulator": self.seed,
                },
            )
            isa_obs = observable.apply_layout(isa.layout)
            groups = len(isa_obs.group_commuting(qubit_wise=True))
            self._cache = (key, isa, isa_obs, groups)
        return self._cache

    def transpiled_metrics(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> dict[str, int]:
        """Depth and two-qubit gate count AFTER transpilation (what actually runs)."""
        _, isa, _, _ = self._prepare(circuit, observable)
        two_q = sum(
            1
            for inst in isa.data
            if len(inst.qubits) == 2 and inst.operation.name not in _TWO_QUBIT_FREE
        )
        return {"transpiled_depth": isa.depth(), "two_qubit_gates": two_q}

    def estimate(
        self, circuit: QuantumCircuit, observable: SparsePauliOp, parameter_values: np.ndarray
    ) -> EstimateResult:
        _, isa, isa_obs, groups = self._prepare(circuit, observable)
        # A fresh but deterministic seed per evaluation: reproducible, yet independent samples.
        assert self._estimator is not None
        self._estimator.options.seed_simulator = self.seed + self._evals
        pub = (isa, isa_obs, np.asarray(parameter_values, dtype=float))
        data = self._estimator.run([pub]).result()[0].data
        self._evals += 1
        self._shots_used += self.shots * groups
        return EstimateResult(
            value=float(data.evs),
            std_error=float(data.stds),
            metadata={"measurement_groups": groups},
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": f"aer_{self.fake_backend_name}",
            "kind": "noisy_simulator",
            "fake_backend": self.fake_backend_name,
            "shots_per_group": self.shots,
            "optimization_level": self.optimization_level,
            "seed": self.seed,
        }

    @property
    def total_shots(self) -> int:
        return self._shots_used
