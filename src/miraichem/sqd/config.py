"""Configuration for one sample-based quantum diagonalization (SQD) run.

SQD is a different workflow from VQE, so it has its own config and result records. It reuses
``MoleculeConfig`` (geometry, basis, active space) and the same hash-keyed caching idea.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from miraichem.config import MoleculeConfig


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

    def config_hash(self) -> str:
        """Stable 12-char hash of the full config (sorted-key JSON, SHA-256)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
