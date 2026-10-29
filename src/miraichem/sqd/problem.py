"""Molecular integrals and the classical references for an SQD run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import comb

import ffsim
import numpy as np
from pyscf import cc, scf

from miraichem.chemistry.active_space import resolve_active_space
from miraichem.chemistry.molecule import build_mole
from miraichem.config import MoleculeConfig

# Exact diagonalisation holds a CI vector over every determinant and takes minutes beyond a few
# million (N2 in 6-31G, 19 million, took 6 minutes), so it is only automatic below this size.
FCI_MAX_DETERMINANTS = 5_000_000


def n_determinants(norb: int, nelec: tuple[int, int]) -> int:
    return comb(norb, nelec[0]) * comb(norb, nelec[1])


@dataclass
class SqdProblem:
    """Everything SQD needs about a molecule. Energies are total energies in Hartree."""

    data: ffsim.MolecularData
    norb: int
    nelec: tuple[int, int]
    hcore: np.ndarray
    eri: np.ndarray
    core_energy: float  # nuclear repulsion plus frozen-core energy
    e_hf: float
    e_exact: float | None  # FCI in the active space (the best SQD can reach); None when too large
    e_ccsd: float  # classical coupled-cluster CCSD in the same active space
    ccsd_t1: np.ndarray
    ccsd_t2: np.ndarray
    fci_time_s: float | None  # how long the exact classical solve took
    ccsd_time_s: float

    @property
    def num_qubits(self) -> int:
        return 2 * self.norb


def build_problem(cfg: MoleculeConfig, bond_length: float, exact: bool | None = None) -> SqdProblem:
    """Run RHF, select the active space, and collect integrals plus the classical references.

    ``exact=None`` computes the exact (FCI) energy only when the space is small enough; pass False
    to skip it. Without it, results are compared with Hartree-Fock and CCSD instead.
    """
    mol = build_mole(cfg, bond_length)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {cfg.name} at {bond_length} A.")
    space = resolve_active_space(cfg, mol)
    indices = None if space is None else list(space.orbital_indices)
    data = ffsim.MolecularData.from_scf(mf, active_space=indices)
    nelec = (data.nelec[0], data.nelec[1])
    if exact is None:
        exact = n_determinants(data.norb, nelec) <= FCI_MAX_DETERMINANTS
    e_exact = None
    fci_time = None
    if exact:
        start = time.perf_counter()
        data.run_fci()
        fci_time = time.perf_counter() - start
        e_exact = float(data.fci_energy)
    # CCSD runs directly on this SCF, so its amplitudes are in the same orbital basis as the
    # integrals and the sampled bitstrings. (ffsim's own run_ccsd rebuilds an SCF that can settle
    # on a different solution, which gave amplitudes that do not belong to this molecule.)
    frozen = None if indices is None else [i for i in range(mol.nao) if i not in set(indices)]
    start = time.perf_counter()
    ccsd = cc.CCSD(mf, frozen=frozen)
    ccsd.kernel()
    ccsd_time = time.perf_counter() - start
    # The integrals must come from this object, not from a separate PySCF run: the sampled
    # bitstrings live in ffsim's choice of molecular orbitals (which differs inside degenerate
    # sets such as N2's pi orbitals), and the integrals have to be in that same basis.
    ham = data.hamiltonian
    return SqdProblem(
        data=data,
        norb=data.norb,
        nelec=nelec,
        hcore=np.asarray(ham.one_body_tensor),
        eri=np.asarray(ham.two_body_tensor),
        core_energy=float(ham.constant),
        e_hf=float(mf.e_tot),
        e_exact=e_exact,
        e_ccsd=float(ccsd.e_tot),
        ccsd_t1=np.asarray(ccsd.t1),
        ccsd_t2=np.asarray(ccsd.t2),
        fci_time_s=fci_time,
        ccsd_time_s=ccsd_time,
    )
