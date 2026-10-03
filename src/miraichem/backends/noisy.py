"""Noisy simulator: Aer with the noise model, gate set and coupling map of a fake IBM backend.

Unlike a plain noiseless simulation, this samples a finite number of shots, applies gate and
readout errors copied from a real device's calibration snapshot, and runs on the transpiled
circuit (routed onto the device's connectivity). It is our stand-in for real hardware during
development, so results here predict how a configuration will behave on a real chip.

We run the measurements ourselves (instead of using a ready-made Estimator) because error
mitigation needs raw counts (readout correction) and control over the circuits (ZNE folding).

How an energy is measured: the Hamiltonian is a sum of Pauli strings. Strings that can be read
out in the same single-qubit measurement basis ("qubit-wise commuting") form a group; one group =
one circuit = one batch of shots. Each term's expectation is the average parity of its qubits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider

from miraichem.backends.base import EstimateResult
from miraichem.mitigation.readout import ReadoutCalibration, calibration_from_counts
from miraichem.mitigation.zne import fold_two_qubit_gates

NoiseKind = Literal["device", "readout_only", "none"]
_NOT_GATES = {"barrier", "delay", "measure"}


def load_fake_backend(name: str):
    """Instantiate a fake IBM backend from ``qiskit_ibm_runtime.fake_provider`` by class name."""
    try:
        return getattr(fake_provider, name)()
    except AttributeError as exc:
        raise ValueError(f"Unknown fake backend {name!r}.") from exc


def _trim_noise_model(model: NoiseModel, qubits: set[int], gate_errors: bool = True) -> NoiseModel:
    """Keep only the errors that act entirely on ``qubits`` (optionally only readout errors).

    Aer re-processes the whole noise model on every call, so a 127-qubit device is ~8x slower
    than a trimmed one even for a 4-qubit circuit. Dropping errors on unused qubits does not change
    the physics of the circuit (idle qubits are discarded by Aer anyway) but makes runs fast.
    Uses NoiseModel's private error tables, which is fine because versions are pinned.
    """
    trimmed = NoiseModel(basis_gates=model.basis_gates)
    if gate_errors:
        for inst, by_qubits in model._local_quantum_errors.items():
            for qs, err in by_qubits.items():
                if set(qs) <= qubits:
                    trimmed.add_quantum_error(err, inst, list(qs))
    for qs, err in model._local_readout_errors.items():
        if set(qs) <= qubits:
            trimmed.add_readout_error(err, list(qs))
    return trimmed


@dataclass
class MeasurementGroup:
    """Pauli terms that share one measurement basis."""

    qubits: list[int]  # physical qubits that are measured; bit j of an outcome = qubits[j]
    basis: dict[int, str]  # physical qubit -> "X" | "Y" | "Z"
    terms: list[tuple[float, list[int]]]  # (coefficient, positions j into ``qubits``)
    values: np.ndarray  # f(outcome), see outcome_values


@dataclass
class GroupSample:
    """Measured outcome distribution of one group (index bit j = result of group.qubits[j])."""

    group: MeasurementGroup
    probs: np.ndarray
    shots: int


@dataclass
class _Prepared:
    isa: QuantumCircuit  # transpiled ansatz (physical qubits), still parameterised
    groups: list[MeasurementGroup]
    constant: float  # coefficient of the identity term
    used_qubits: list[int]
    templates: dict[float, list[QuantumCircuit]]  # ZNE scale -> one circuit per group


def _basis_change(circuit: QuantumCircuit, qubit: int, basis: str) -> None:
    """Append native gates (rz, sx) that rotate ``basis`` onto Z. H = rz(pi/2) sx rz(pi/2)."""
    if basis == "X":
        circuit.rz(np.pi / 2, qubit)
        circuit.sx(qubit)
        circuit.rz(np.pi / 2, qubit)
    elif basis == "Y":  # Sdg then H, which simplifies to sx then rz(pi/2)
        circuit.sx(qubit)
        circuit.rz(np.pi / 2, qubit)


def _build_groups(observable: SparsePauliOp) -> tuple[list[MeasurementGroup], float]:
    """Split a (physical-qubit) Pauli operator into measurement groups plus a constant."""
    constant = 0.0
    groups: list[MeasurementGroup] = []
    for sub in observable.group_commuting(qubit_wise=True):
        basis: dict[int, str] = {}
        raw_terms: list[tuple[float, list[int]]] = []
        for pauli, coeff in zip(sub.paulis, sub.coeffs, strict=True):
            support = [q for q in range(pauli.num_qubits) if pauli.x[q] or pauli.z[q]]
            if not support:
                constant += float(np.real(coeff))
                continue
            for q in support:
                basis[q] = "Y" if (pauli.x[q] and pauli.z[q]) else ("X" if pauli.x[q] else "Z")
            raw_terms.append((float(np.real(coeff)), support))
        if not raw_terms:
            continue
        qubits = sorted(basis)
        terms = [(c, [qubits.index(q) for q in s]) for c, s in raw_terms]
        groups.append(MeasurementGroup(qubits, basis, terms, outcome_values(terms, len(qubits))))
    return groups, constant


def outcome_values(terms: list[tuple[float, list[int]]], num_bits: int) -> np.ndarray:
    """f(outcome) = sum over terms of coeff * parity of the term's bits, for every outcome.

    A Pauli string measured in the Z basis equals +1 for an even number of 1s on its qubits, -1
    for an odd number.
    """
    idx = np.arange(2**num_bits)
    f = np.zeros(2**num_bits)
    for coeff, pos in terms:
        mask = sum(1 << j for j in pos)
        ones = np.array([bin(i & mask).count("1") for i in idx])
        f += coeff * (1 - 2 * (ones % 2))
    return f


def group_value(sample: GroupSample, probs: np.ndarray | None = None) -> tuple[float, float]:
    """Expectation value (and variance of the mean) of one group from an outcome distribution.

    ``probs`` lets mitigation substitute a corrected distribution; the variance always comes from
    the raw sampled distribution (an approximation when mitigation is on).
    """
    f = sample.group.values
    raw_mean = float(sample.probs @ f)
    var = max(float(sample.probs @ f**2) - raw_mean**2, 0.0) / sample.shots
    mean = raw_mean if probs is None else float(probs @ f)
    return mean, var


def energy_from_samples(
    samples: list[GroupSample],
    constant: float,
    readout: ReadoutCalibration | None = None,
) -> tuple[float, float]:
    """Sum group expectation values (optionally readout-corrected). Returns (value, variance)."""
    total, var = constant, 0.0
    for s in samples:
        probs = readout.correct(s.probs, s.group.qubits) if readout else None
        mean, v = group_value(s, probs)
        total += mean
        var += v
    return total, var


class NoisyBackend:
    """Shot-based energy estimator on an Aer simulator that mimics a named fake IBM backend."""

    def __init__(
        self,
        fake_backend_name: str = "FakeSherbrooke",
        shots: int = 4096,
        seed: int = 42,
        optimization_level: int = 3,
        noise: NoiseKind = "device",
    ) -> None:
        self.fake_backend_name = fake_backend_name
        self.shots = shots
        self.seed = seed
        self.optimization_level = optimization_level
        self.noise = noise
        self._fake = load_fake_backend(fake_backend_name)
        # from_backend copies the noise model, basis gates and coupling map of the device.
        self._sim = AerSimulator.from_backend(self._fake, seed_simulator=seed)
        self._run_sim: AerSimulator | None = None
        self._prepared: tuple[int, _Prepared] | None = None
        self._calibration: ReadoutCalibration | None = None
        self._runs = 0
        self._shots_used = 0

    # -- preparation ---------------------------------------------------------------------------

    def prepare(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> _Prepared:
        """Transpile once to the device and split the observable into measurement groups."""
        if self._prepared is not None and self._prepared[0] == id(circuit):
            return self._prepared[1]
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            backend=self._sim,
            seed_transpiler=self.seed,
        )
        isa = pm.run(circuit)
        isa_obs = observable.apply_layout(isa.layout)
        groups, constant = _build_groups(isa_obs)
        used = sorted({isa.find_bit(q).index for inst in isa.data for q in inst.qubits})
        self._run_sim = self._make_run_sim(set(used))
        prepared = _Prepared(isa, groups, constant, used, {})
        self._prepared = (id(circuit), prepared)
        self._calibration = None
        return prepared

    def _make_run_sim(self, qubits: set[int]) -> AerSimulator:
        if self.noise == "none":
            empty = NoiseModel(basis_gates=self._sim.options.noise_model.basis_gates)
            return AerSimulator.from_backend(
                self._fake, noise_model=empty, seed_simulator=self.seed
            )
        model = _trim_noise_model(
            self._sim.options.noise_model, qubits, gate_errors=self.noise == "device"
        )
        return AerSimulator.from_backend(self._fake, noise_model=model, seed_simulator=self.seed)

    def templates(self, prepared: _Prepared, scale: float) -> list[QuantumCircuit]:
        """One parameterised measurement circuit per group, at ZNE noise scale ``scale``."""
        if scale not in prepared.templates:
            base = fold_two_qubit_gates(prepared.isa, scale) if scale != 1.0 else prepared.isa
            circuits = []
            for g in prepared.groups:
                qc = base.copy()
                creg = ClassicalRegister(len(g.qubits), "m")
                qc.add_register(creg)
                for q in g.qubits:
                    _basis_change(qc, q, g.basis[q])
                for j, q in enumerate(g.qubits):
                    qc.measure(q, creg[j])
                circuits.append(qc)
            prepared.templates[scale] = circuits
        return prepared.templates[scale]

    # -- execution -----------------------------------------------------------------------------

    def execute(self, circuits: list[QuantumCircuit]) -> list[dict[str, int]]:
        """Run circuits (already in the device gate set) and return counts for each."""
        assert self._run_sim is not None, "call prepare() first"
        seed = self.seed + self._runs
        self._runs += 1
        result = self._run_sim.run(circuits, shots=self.shots, seed_simulator=seed).result()
        self._shots_used += self.shots * len(circuits)
        return [result.get_counts(i) for i in range(len(circuits))]

    def sample(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        parameter_values: np.ndarray,
        scale: float = 1.0,
    ) -> list[GroupSample]:
        """Measure every group once at noise scale ``scale``; returns outcome distributions."""
        prepared = self.prepare(circuit, observable)
        values = np.asarray(parameter_values, dtype=float)
        bound = [qc.assign_parameters(values) for qc in self.templates(prepared, scale)]
        samples = []
        for group, counts in zip(prepared.groups, self.execute(bound), strict=True):
            probs = np.zeros(2 ** len(group.qubits))
            for key, n in counts.items():
                probs[int(key.replace(" ", ""), 2)] += n
            samples.append(GroupSample(group, probs / self.shots, self.shots))
        return samples

    def readout_calibration(self, circuit: QuantumCircuit, observable: SparsePauliOp):
        """Measure per-qubit readout errors with all-|0> and all-|1> circuits (cached)."""
        if self._calibration is None:
            prepared = self.prepare(circuit, observable)
            qubits = prepared.used_qubits
            circuits = []
            for flip in (False, True):
                qc = QuantumCircuit(prepared.isa.num_qubits, len(qubits))
                if flip:
                    for q in qubits:
                        qc.x(q)
                for j, q in enumerate(qubits):
                    qc.measure(q, j)
                circuits.append(qc)
            zero, one = self.execute(circuits)
            self._calibration = calibration_from_counts(
                {k.replace(" ", ""): v for k, v in zero.items()},
                {k.replace(" ", ""): v for k, v in one.items()},
                qubits,
            )
        return self._calibration

    # -- EnergyBackend interface -----------------------------------------------------------------

    def transpiled_metrics(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> dict[str, int]:
        """Depth and two-qubit gate count AFTER transpilation (what actually runs)."""
        isa = self.prepare(circuit, observable).isa
        two_q = sum(
            1
            for inst in isa.data
            if len(inst.qubits) == 2 and inst.operation.name not in _NOT_GATES
        )
        return {"transpiled_depth": isa.depth(), "two_qubit_gates": two_q}

    def estimate(
        self, circuit: QuantumCircuit, observable: SparsePauliOp, parameter_values: np.ndarray
    ) -> EstimateResult:
        """Unmitigated energy expectation value."""
        prepared = self.prepare(circuit, observable)
        samples = self.sample(circuit, observable, parameter_values)
        value, var = energy_from_samples(samples, prepared.constant)
        return EstimateResult(
            value=value,
            std_error=float(np.sqrt(var)),
            metadata={"measurement_groups": len(prepared.groups)},
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": f"aer_{self.fake_backend_name}",
            "kind": "noisy_simulator",
            "fake_backend": self.fake_backend_name,
            "shots_per_group": self.shots,
            "optimization_level": self.optimization_level,
            "noise": self.noise,
            "seed": self.seed,
        }

    @property
    def total_shots(self) -> int:
        return self._shots_used
