"""Classical reference energies (Hartree-Fock and exact) from PySCF.

All energies returned here are TOTAL energies in Hartree: electronic energy plus nuclear
repulsion (and, for an active space, the frozen-core energy). That is the same quantity the
qubit Hamiltonian reproduces once its constant ``energy_shift`` is added, so numbers can be
compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyscf import fci, mcscf, scf

from miraichem.chemistry.active_space import resolve_active_space
from miraichem.chemistry.molecule import build_mole
from miraichem.config import MoleculeConfig


@dataclass(frozen=True)
class ClassicalReference:
    """Reference energies in Hartree (total, nuclear repulsion included)."""

    e_hf: float
    e_exact: float
    e_nuclear_repulsion: float
    n_qubits_expected: int  # Jordan-Wigner qubit count (2 per active spatial orbital)
    exact_method: str  # "FCI" (full space) or "CASCI" (active space)


def compute_reference(cfg: MoleculeConfig, bond_length: float) -> ClassicalReference:
    """HF and exact (FCI, or CASCI inside the active space) total energies.

    CASCI = exact diagonalisation of the Hamiltonian restricted to the active space, i.e. the
    best energy any VQE in that same active space could possibly reach.
    """
    mol = build_mole(cfg, bond_length)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {cfg.name} at {bond_length} A.")

    space = resolve_active_space(cfg, mol)
    if space is None:
        solver = fci.FCI(mf)
        solver.conv_tol = 1e-12
        e_exact = float(solver.kernel()[0])
        method, n_orb = "FCI", mol.nao
    else:
        mc = mcscf.CASCI(mf, space.num_orbitals, space.num_electrons)
        mc.fcisolver.conv_tol = 1e-12
        # sort_mo takes 1-based indices of the MOs to place in the active window.
        mc.mo_coeff = mc.sort_mo([i + 1 for i in space.orbital_indices])
        e_exact = float(mc.kernel()[0])
        method, n_orb = "CASCI", space.num_orbitals

    return ClassicalReference(
        e_hf=float(mf.e_tot),
        e_exact=e_exact,
        e_nuclear_repulsion=float(mol.energy_nuc()),
        n_qubits_expected=2 * n_orb,
        exact_method=method,
    )
