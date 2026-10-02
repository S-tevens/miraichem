import pytest
from pydantic import ValidationError

from miraichem.config import RunConfig, SweepConfig


def test_hash_is_stable_and_distinguishes_configs(h2):
    a = RunConfig(molecule=h2, bond_length=0.735)
    b = RunConfig(molecule=h2, bond_length=0.735)
    c = RunConfig(molecule=h2, bond_length=0.8)
    d = RunConfig(molecule=h2, bond_length=0.735, seed=7)
    assert a.config_hash() == b.config_hash()
    assert len(a.config_hash()) == 12
    assert len({a.config_hash(), c.config_hash(), d.config_hash()}) == 3


def test_invalid_combinations_rejected(h2):
    with pytest.raises(ValidationError):
        RunConfig(molecule=h2, bond_length=0.735, optimizer="lbfgsb", backend="noisy")
    with pytest.raises(ValidationError):
        RunConfig(molecule=h2, bond_length=0.735, backend="ideal", mitigation="zne")


def test_active_space_validation(h2):
    data = h2.model_dump()
    data["active_electrons"] = 2
    with pytest.raises(ValidationError):
        type(h2)(**data)  # active_orbitals missing
    data = h2.model_dump()
    data["spin"] = 2
    with pytest.raises(ValidationError):
        type(h2)(**data)


def test_sweep_expansion_skips_invalid_and_excluded(h2):
    sweep = SweepConfig(
        molecule=h2,
        bond_lengths=[0.735],
        optimizers=["cobyla", "lbfgsb"],
        backends=["ideal", "noisy"],
        mitigations=["none", "readout"],
        exclude=[{"optimizer": "cobyla", "backend": "ideal"}],
    )
    runs = sweep.expand()
    combos = {(r.optimizer, r.backend, r.mitigation) for r in runs}
    assert combos == {
        ("cobyla", "noisy", "none"),
        ("cobyla", "noisy", "readout"),
        ("lbfgsb", "ideal", "none"),
    }
