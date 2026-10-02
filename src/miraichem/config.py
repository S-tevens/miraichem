"""Pydantic configuration models for molecules, single runs and sweeps.

Everything the pipeline does is described by these models, so a run is fully reproducible from its
config. ``RunConfig.config_hash()`` is the key used for caching results and naming result files.
Geometry is always in Angstrom; all energies in the project are total energies in Hartree.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Mapping = Literal["jordan_wigner", "parity"]
AnsatzKind = Literal["hea", "uccsd"]
OptimizerKind = Literal["cobyla", "spsa", "lbfgsb"]
BackendKind = Literal["ideal", "noisy", "hardware"]
MitigationKind = Literal["none", "readout", "zne", "readout+zne"]


class MoleculeConfig(BaseModel):
    """A molecule plus the (optional) active space used to shrink it to a few qubits.

    ``geometry`` is a PySCF-style atom string in Angstrom. It may contain a ``{bond_length}``
    placeholder so the same config can be scanned over bond lengths.

    The active space freezes the lowest orbitals (their electrons form a constant "core energy")
    and keeps only a few orbitals/electrons for the quantum computer. Fewer orbitals means fewer
    qubits, which is what makes LiH feasible on a laptop and on noisy hardware.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    geometry: str
    basis: str = "sto3g"
    charge: int = 0
    spin: int = Field(0, description="2S = number of unpaired electrons")
    active_electrons: int | None = None
    active_orbitals: int | list[int] | None = Field(
        None,
        description="Number of active spatial orbitals, or an explicit list of 0-based MO indices.",
    )
    freeze_core: bool = False

    @field_validator("spin")
    @classmethod
    def _closed_shell_only(cls, v: int) -> int:
        if v != 0:
            raise ValueError("Only closed-shell molecules (spin=0) are supported in the MVP.")
        return v

    @model_validator(mode="after")
    def _check_active_space(self) -> MoleculeConfig:
        if (self.active_electrons is None) != (self.active_orbitals is None):
            raise ValueError("Set both active_electrons and active_orbitals, or neither.")
        if self.active_electrons is not None and self.freeze_core:
            raise ValueError("Use either an explicit active space or freeze_core, not both.")
        if isinstance(self.active_orbitals, list):
            n = len(set(self.active_orbitals))
            if n != len(self.active_orbitals):
                raise ValueError("active_orbitals contains duplicates.")
        return self

    def atom_string(self, bond_length: float) -> str:
        """Geometry string with ``{bond_length}`` substituted (Angstrom)."""
        return self.geometry.format(bond_length=bond_length)


class RunConfig(BaseModel):
    """One VQE run: a molecule at one bond length with one full set of algorithm choices."""

    model_config = ConfigDict(extra="forbid")

    molecule: MoleculeConfig
    bond_length: float = Field(gt=0, description="Angstrom")
    mapping: Mapping = "jordan_wigner"
    ansatz: AnsatzKind = "uccsd"
    ansatz_reps: int = Field(1, ge=1)
    optimizer: OptimizerKind = "cobyla"
    maxiter: int = Field(200, ge=1)
    backend: BackendKind = "ideal"
    fake_backend_name: str = "FakeSherbrooke"
    shots: int = Field(4096, ge=1)
    mitigation: MitigationKind = "none"
    seed: int = 42

    @model_validator(mode="after")
    def _check_combo(self) -> RunConfig:
        reason = invalid_combination(self.optimizer, self.backend, self.mitigation)
        if reason:
            raise ValueError(reason)
        return self

    def config_hash(self) -> str:
        """Stable 12-char hash of the full config (sorted-key JSON, SHA-256)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def invalid_combination(optimizer: str, backend: str, mitigation: str) -> str | None:
    """Return why a combination is invalid, or None if it is fine."""
    if optimizer == "lbfgsb" and backend != "ideal":
        return "lbfgsb needs exact values and gradients: use it with the ideal backend only."
    if mitigation != "none" and backend == "ideal":
        return "Error mitigation only makes sense on noisy or hardware backends."
    return None


class SweepConfig(BaseModel):
    """A grid of runs. Expands to the Cartesian product of every list, minus invalid combos."""

    model_config = ConfigDict(extra="forbid")

    molecule: MoleculeConfig
    bond_lengths: list[float] = Field(min_length=1)
    mappings: list[Mapping] = ["jordan_wigner"]
    ansatzes: list[AnsatzKind] = ["uccsd"]
    ansatz_reps: list[int] = [1]
    optimizers: list[OptimizerKind] = ["cobyla"]
    maxiter: int = 200
    backends: list[BackendKind] = ["ideal"]
    fake_backend_name: str = "FakeSherbrooke"
    shots: list[int] = [4096]
    mitigations: list[MitigationKind] = ["none"]
    seed: int = 42
    exclude: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rules: a combo is dropped if it matches ALL key/value pairs of any rule.",
    )

    def expand(self) -> list[RunConfig]:
        """All valid RunConfigs, with invalid combinations and ``exclude`` matches skipped."""
        runs: list[RunConfig] = []
        grid = itertools.product(
            self.bond_lengths,
            self.mappings,
            self.ansatzes,
            self.ansatz_reps,
            self.optimizers,
            self.backends,
            self.shots,
            self.mitigations,
        )
        for bl, mp, an, reps, opt, be, sh, mit in grid:
            if invalid_combination(opt, be, mit):
                continue
            fields: dict[str, Any] = dict(
                molecule=self.molecule,
                bond_length=bl,
                mapping=mp,
                ansatz=an,
                ansatz_reps=reps,
                optimizer=opt,
                maxiter=self.maxiter,
                backend=be,
                fake_backend_name=self.fake_backend_name,
                shots=sh,
                mitigation=mit,
                seed=self.seed,
            )
            if any(all(fields.get(k) == v for k, v in rule.items()) for rule in self.exclude):
                continue
            runs.append(RunConfig(**fields))
        return runs
