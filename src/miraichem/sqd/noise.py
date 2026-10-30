"""A device-calibrated noise model for simulated SQD sampling.

The plain simulator flips every bit with one probability, which is not what a device does. This
model takes the calibration of a (fake) IBM device and the circuit as it would actually be
compiled for it, and applies two effects:

  * readout error: each measured bit flips with the readout error of the physical qubit it was
    measured on, so some bits are noisier than others;
  * gate error: the chance that a shot runs through the whole circuit untouched is the product of
    ``1 - error`` over every gate. A shot that does not survive is replaced by a uniformly random
    bitstring (a global depolarising approximation).

It is an approximation: it does not simulate the noisy circuit gate by gate, and it ignores
crosstalk, drift and leakage. It is more realistic than a uniform bit flip, but it is still not a
real device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miraichem.backends.noisy import load_fake_backend
from miraichem.sqd.problem import SqdProblem
from miraichem.sqd.sampler import device_interaction_pairs  # noqa: F401  (re-exported for callers)

_NOT_GATES = {"measure", "barrier", "delay"}


@dataclass
class DeviceNoise:
    backend: str
    readout_error: np.ndarray  # per bitstring column, in sampled-string order
    survival: float  # probability a shot is untouched by gate error
    two_qubit_gates: int

    def apply(self, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return noisy copies of the sampled ``bits`` (shots x qubits, bool)."""
        out = bits.copy()
        scrambled = rng.random(len(out)) > self.survival
        out[scrambled] = rng.random((int(scrambled.sum()), out.shape[1])) < 0.5
        out ^= rng.random(out.shape) < self.readout_error
        return out

    def summary(self) -> dict[str, float | int | str]:
        return {
            "backend": self.backend,
            "survival": float(self.survival),
            "two_qubit_gates": self.two_qubit_gates,
            "median_readout_error": float(np.median(self.readout_error)),
        }


def device_noise(problem: SqdProblem, n_reps: int, backend_name: str) -> DeviceNoise:
    """Compile the device-friendly circuit for ``backend_name`` and read off its error rates."""
    # Imported here: hardware imports the pipeline, which imports this module.
    from miraichem.sqd.hardware import build_circuit, transpile_for

    device = load_fake_backend(backend_name)
    isa = transpile_for(build_circuit(problem, n_reps), device)
    target = device.target

    def error_of(name: str, qubits: tuple[int, ...]) -> float:
        try:
            props = target[name][qubits]
        except (KeyError, TypeError):
            return 0.0
        err = getattr(props, "error", None)
        return float(err) if err else 0.0

    survival = 1.0
    two_qubit = 0
    readout = np.zeros(problem.num_qubits)
    for inst in isa.data:
        name = inst.operation.name
        qubits = tuple(isa.find_bit(q).index for q in inst.qubits)
        if name == "measure":
            clbit = isa.find_bit(inst.clbits[0]).index
            readout[clbit] = error_of("measure", qubits)
        elif name not in _NOT_GATES:
            survival *= 1.0 - error_of(name, qubits)
            two_qubit += len(qubits) == 2
    # Sampled strings list the highest qubit first, so reverse to line columns up with qubits.
    return DeviceNoise(device.name, readout[::-1].copy(), survival, two_qubit)
