"""Active-space selection.

An active space keeps only a few molecular orbitals (MOs) and electrons for the quantum
computer. Orbitals below it are "frozen" (always doubly occupied) and contribute a constant
core energy; orbitals above it are dropped. The same active space must be used for the classical
CASCI reference and for the qubit Hamiltonian, otherwise energy comparisons are meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyscf import gto

from miraichem.config import MoleculeConfig

# Number of core (1s, 2s2p, ...) spatial orbitals frozen per atom, by atomic number.
_CORE_BY_Z = [(2, 0), (10, 1), (18, 5)]


@dataclass(frozen=True)
class ActiveSpace:
    """0-based MO indices (canonical RHF order) and electron count kept active."""

    num_electrons: int
    orbital_indices: tuple[int, ...]

    @property
    def num_orbitals(self) -> int:
        return len(self.orbital_indices)


def _core_orbitals(mol: gto.Mole) -> int:
    total = 0
    for z in mol.atom_charges():
        for zmax, ncore in _CORE_BY_Z:
            if z <= zmax:
                total += ncore
                break
        else:
            raise ValueError(f"freeze_core not supported for atomic number {z}.")
    return total


def resolve_active_space(cfg: MoleculeConfig, mol: gto.Mole) -> ActiveSpace | None:
    """Turn the config into explicit MO indices, or None if the full space is used."""
    nelec = mol.nelectron
    nmo = mol.nao
    if cfg.freeze_core:
        ncore = _core_orbitals(mol)
        return ActiveSpace(nelec - 2 * ncore, tuple(range(ncore, nmo)))
    if cfg.active_electrons is None:
        return None
    act = cfg.active_orbitals
    if isinstance(act, int):
        # Orbitals centred on the HOMO-LUMO gap, like qiskit-nature's default.
        n_act_occ = cfg.active_electrons // 2
        ncore = (nelec - cfg.active_electrons) // 2
        indices = tuple(range(ncore, ncore + act))
        if n_act_occ > act:
            raise ValueError("More active electron pairs than active orbitals.")
    else:
        indices = tuple(sorted(act))  # type: ignore[arg-type]
    if max(indices) >= nmo or min(indices) < 0:
        raise ValueError(f"Active orbital index out of range (molecule has {nmo} orbitals).")
    if cfg.active_electrons > 2 * len(indices):
        raise ValueError("More active electrons than the active orbitals can hold.")
    if (nelec - cfg.active_electrons) % 2:
        raise ValueError("Frozen electrons must pair up (closed-shell core).")
    return ActiveSpace(cfg.active_electrons, indices)
