"""Build PySCF molecules from a MoleculeConfig."""

from __future__ import annotations

from pyscf import gto

from miraichem.config import MoleculeConfig


def build_mole(cfg: MoleculeConfig, bond_length: float) -> gto.Mole:
    """Return a PySCF ``Mole`` (geometry in Angstrom) for ``cfg`` at ``bond_length``."""
    return gto.M(
        atom=cfg.atom_string(bond_length),
        basis=cfg.basis,
        charge=cfg.charge,
        spin=cfg.spin,
        unit="Angstrom",
        verbose=0,
    )
