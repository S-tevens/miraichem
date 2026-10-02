from pathlib import Path

import pytest
import yaml

from miraichem.config import MoleculeConfig

ROOT = Path(__file__).resolve().parents[1]


def load_molecule(name: str) -> MoleculeConfig:
    with open(ROOT / "configs" / "molecules" / f"{name}.yaml") as f:
        return MoleculeConfig(**yaml.safe_load(f))


@pytest.fixture(scope="session")
def h2() -> MoleculeConfig:
    return load_molecule("h2")


@pytest.fixture(scope="session")
def lih() -> MoleculeConfig:
    return load_molecule("lih")
