"""Mitigated energy backend: wraps a NoisyBackend with readout correction and/or ZNE.

Every call returns the mitigated energy as ``value`` and ALSO the raw unmitigated energy in
``metadata['unmitigated']``, so "before vs after" can be plotted for every evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from miraichem.backends.base import EstimateResult
from miraichem.backends.noisy import NoisyBackend, energy_from_samples
from miraichem.config import MitigationKind
from miraichem.mitigation.zne import Method, extrapolate


class MitigatedBackend:
    """Applies readout mitigation, ZNE, or both on top of a noisy simulator."""

    def __init__(
        self,
        base: NoisyBackend,
        mitigation: MitigationKind,
        zne_scales: list[float] | None = None,
        zne_method: Method = "linear",
    ) -> None:
        if mitigation == "none":
            raise ValueError("Use the NoisyBackend directly when no mitigation is requested.")
        self.base = base
        self.use_readout = "readout" in mitigation
        self.use_zne = "zne" in mitigation
        self.mitigation = mitigation
        self.scales = list(zne_scales or [1.0, 3.0])
        self.zne_method = zne_method
        if self.use_zne and (len(self.scales) < 2 or self.scales[0] != 1.0):
            raise ValueError("ZNE needs at least two scale factors, the first being 1.")

    def estimate(
        self, circuit: QuantumCircuit, observable: SparsePauliOp, parameter_values: np.ndarray
    ) -> EstimateResult:
        prepared = self.base.prepare(circuit, observable)
        readout = self.base.readout_calibration(circuit, observable) if self.use_readout else None
        scales = self.scales if self.use_zne else [1.0]
        raw: list[tuple[float, float]] = []
        mitigated: list[tuple[float, float]] = []
        for scale in scales:
            samples = self.base.sample(circuit, observable, parameter_values, scale)
            raw.append(energy_from_samples(samples, prepared.constant))
            mitigated.append(energy_from_samples(samples, prepared.constant, readout))

        if self.use_zne:
            value, std = extrapolate(
                scales, [m[0] for m in mitigated], [m[1] for m in mitigated], self.zne_method
            )
        else:
            value, std = mitigated[0][0], float(np.sqrt(mitigated[0][1]))
        meta: dict[str, Any] = {
            "unmitigated": raw[0][0],
            "mitigation": self.mitigation,
            "scales": scales,
            "values_by_scale": [m[0] for m in mitigated],
            "measurement_groups": len(prepared.groups),
        }
        return EstimateResult(value=value, std_error=std, metadata=meta)

    def transpiled_metrics(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> dict[str, int]:
        return self.base.transpiled_metrics(circuit, observable)

    def info(self) -> dict[str, Any]:
        info = self.base.info()
        info.update(
            mitigation=self.mitigation,
            zne_scales=self.scales if self.use_zne else None,
            zne_method=self.zne_method if self.use_zne else None,
        )
        return info

    @property
    def total_shots(self) -> int:
        return self.base.total_shots
