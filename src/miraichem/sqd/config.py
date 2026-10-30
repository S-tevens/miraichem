"""Configuration for one sample-based quantum diagonalization (SQD) run.

SQD is a different workflow from VQE, so it has its own config and result records. It reuses
``MoleculeConfig`` (geometry, basis, active space) and the same hash-keyed caching idea.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miraichem.config import MoleculeConfig

DEFAULT_BOND_LENGTH = 1.0977  # Angstrom (N2); other molecules set bond_length in their YAML


def load_sqd_yaml(text: str) -> tuple[MoleculeConfig, float]:
    """Read an SQD molecule file: the molecule plus its default bond length in Angstrom.

    ``bond_length`` is an extra key in the YAML (not part of ``MoleculeConfig``, so that adding it
    does not change any result's hash).
    """
    data = yaml.safe_load(text)
    bond = float(data.pop("bond_length", DEFAULT_BOND_LENGTH))
    return MoleculeConfig(**data), bond


class SqdConfig(BaseModel):
    """One SQD run: a molecule plus how bitstrings are sampled and how the subspace is built.

    Samples come from a local-unitary cluster Jastrow (LUCJ) circuit applied to the Hartree-Fock
    state, simulated exactly with ffsim. ``readout_noise`` flips each sampled bit independently
    with that probability, a simple stand-in for measurement noise (it is NOT a device model).
    Without noise the ideal LUCJ state concentrates on a few dozen bitstrings, which is too few to
    span a useful subspace; noise adds diversity, so use a small non-zero ``readout_noise`` for
    a meaningful simulated run.
    """

    model_config = ConfigDict(extra="forbid")

    molecule: MoleculeConfig
    bond_length: float = Field(gt=0, description="Angstrom")
    n_reps: int = Field(1, ge=1, description="LUCJ layers")
    shots: int = Field(20000, ge=1)
    readout_noise: float = Field(0.0, ge=0, lt=0.5, description="bit-flip probability per bit")
    iterations: int = Field(3, ge=1, description="SQD rounds; rounds after the first use recovery")
    samples_per_batch: int = Field(300, ge=1, description="bitstrings per subspace batch")
    num_batches: int = Field(3, ge=1, description="independent subspaces per round")
    seed: int = 42
    noise_model: str | None = Field(
        None,
        description=(
            "Name of a fake IBM device (for example FakeFez) whose calibration drives the sampling "
            "noise instead of the uniform bit flip; leave readout_noise at 0 when it is set"
        ),
    )

    sampler: Literal["circuit", "hf_noise", "uniform"] = Field(
        "circuit",
        description=(
            "Where the bitstrings come from. 'circuit' samples the LUCJ circuit (the method). The "
            "other two are controls that leave the circuit out: 'hf_noise' samples only the "
            "Hartree-Fock state and adds the same bit-flip noise, 'uniform' draws every bit at "
            "random. They show what the circuit adds on top of noise and recovery."
        ),
    )

    @model_validator(mode="after")
    def _one_noise_source(self) -> SqdConfig:
        if self.noise_model and self.readout_noise > 0:
            raise ValueError("Use either readout_noise or noise_model, not both.")
        if self.sampler != "circuit" and self.noise_model:
            raise ValueError("A control sampler uses readout_noise, not a device noise model.")
        return self

    def config_hash(self) -> str:
        """Stable 12-char hash of the full config (sorted-key JSON, SHA-256).

        A field added later is left out while it is unset, so results saved before it existed keep
        their hash.
        """
        data = self.model_dump(mode="json")
        if data["noise_model"] is None:
            del data["noise_model"]
        if data["sampler"] == "circuit":
            del data["sampler"]
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
