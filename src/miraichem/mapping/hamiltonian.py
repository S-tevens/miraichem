"""Molecule -> qubit Hamiltonian, plus an exact-diagonalisation check.

The qubit Hamiltonian only contains the electronic part acting on the active orbitals. Constants
(nuclear repulsion, frozen-core energy) are returned separately as ``energy_shift`` so that
``eigenvalue + energy_shift`` is the TOTAL energy, comparable to PySCF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

from miraichem.chemistry.active_space import resolve_active_space
from miraichem.chemistry.molecule import build_mole
from miraichem.config import Mapping, MoleculeConfig
from miraichem.mapping.mappers import get_mapper


@dataclass
class QubitProblem:
    """Everything downstream code needs about the qubit problem.

    ``mapper`` is kept so the Hartree-Fock initial state and UCCSD ansatz can be built with the
    same mapping (``HartreeFock(num_spatial_orbitals, num_particles, mapper)``).
    """

    hamiltonian: SparsePauliOp
    energy_shift: float  # Hartree: nuclear repulsion + frozen-core energy
    num_qubits: int
    num_particles: tuple[int, int]  # (n_alpha, n_beta) in the active space
    num_spatial_orbitals: int
    mapping: Mapping
    mapper: QubitMapper


def build_qubit_hamiltonian(
    cfg: MoleculeConfig, bond_length: float, mapping: Mapping = "jordan_wigner"
) -> QubitProblem:
    """Build the qubit Hamiltonian for ``cfg`` at ``bond_length`` (Angstrom)."""
    mol = build_mole(cfg, bond_length)
    driver = PySCFDriver(
        atom=cfg.atom_string(bond_length),
        basis=cfg.basis,
        charge=cfg.charge,
        spin=cfg.spin,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    space = resolve_active_space(cfg, mol)
    if space is not None:
        transformer = ActiveSpaceTransformer(
            space.num_electrons,
            space.num_orbitals,
            active_orbitals=list(space.orbital_indices),
        )
        problem = transformer.transform(problem)  # type: ignore[assignment]

    ham = problem.hamiltonian
    assert problem.num_particles is not None and problem.num_spatial_orbitals is not None
    num_particles = (int(problem.num_particles[0]), int(problem.num_particles[1]))
    mapper = get_mapper(mapping, num_particles)
    qubit_op = mapper.map(ham.second_q_op())
    shift = float(sum(ham.constants.values()))
    return QubitProblem(
        hamiltonian=qubit_op.simplify(),
        energy_shift=shift,
        num_qubits=qubit_op.num_qubits,
        num_particles=num_particles,
        num_spatial_orbitals=int(problem.num_spatial_orbitals),
        mapping=mapping,
        mapper=mapper,
    )


def exact_ground_energy(problem: QubitProblem, penalty: float = 50.0) -> float:
    """Exact lowest TOTAL energy of the qubit Hamiltonian, in the correct electron-number sector.

    The full qubit Hamiltonian also has eigenstates with the wrong number of electrons (some of
    them lower in energy). We add a large penalty ``penalty * (N - N_target)^2`` so only the right
    sector can be lowest, then verify the winning state really has N_target electrons. The penalty
    is built with the same mapper so this works for Jordan-Wigner and Parity alike.
    """
    n_orb = 2 * problem.num_spatial_orbitals
    n_target = sum(problem.num_particles)
    number_op = FermionicOp({f"+_{i} -_{i}": 1.0 for i in range(n_orb)}, num_spin_orbitals=n_orb)
    n_qubit = problem.mapper.map(number_op)
    n_mat = n_qubit.to_matrix()
    h_mat = problem.hamiltonian.to_matrix()
    dev = n_mat - n_target * np.eye(n_mat.shape[0])
    values, vectors = np.linalg.eigh(h_mat + penalty * (dev @ dev))
    ground = vectors[:, 0]
    n_expect = float(np.real(ground.conj() @ n_mat @ ground))
    if abs(n_expect - n_target) > 1e-6:
        raise RuntimeError(f"Ground state has {n_expect:.4f} electrons, expected {n_target}.")
    energy = float(np.real(ground.conj() @ h_mat @ ground))
    return energy + problem.energy_shift
