"""Phase 0 spike: H2 / STO-3G at 0.735 A.

Computes HF and FCI with PySCF, then builds the qubit Hamiltonian two ways
(qiskit-nature, openfermion) and checks that its exact ground-state energy matches FCI.
Exact diagonalisation is restricted to the 2-electron sector because the full qubit
Hamiltonian also contains states with the wrong particle number.
"""
import numpy as np
from pyscf import fci, gto, scf

BOND = 0.735  # Angstrom
ATOM = f"H 0 0 0; H 0 0 {BOND}"

mol = gto.M(atom=ATOM, basis="sto3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol).run()
e_fci = fci.FCI(mf).kernel()[0]
print(f"PySCF HF  : {mf.e_tot:.8f} Ha")
print(f"PySCF FCI : {e_fci:.8f} Ha")


def sector_ground(h_matrix, n_qubits, n_elec):
    """Lowest eigenvalue among computational basis states with n_elec electrons."""
    idx = [i for i in range(2**n_qubits) if bin(i).count("1") == n_elec]
    return np.linalg.eigvalsh(h_matrix[np.ix_(idx, idx)])[0]


# Route 1: qiskit-nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

problem = PySCFDriver(atom=ATOM, basis="sto3g", unit=__import__("qiskit_nature.units", fromlist=["DistanceUnit"]).DistanceUnit.ANGSTROM).run()
mapper = JordanWignerMapper()
op = mapper.map(problem.hamiltonian.second_q_op())
shift = sum(problem.hamiltonian.constants.values())
e_qn = sector_ground(op.to_matrix(), op.num_qubits, 2) + shift
print(f"qiskit-nature JW ({op.num_qubits} qubits): {e_qn:.8f} Ha  (diff {abs(e_qn - e_fci):.2e})")

# Route 2: openfermion
from openfermion import MolecularData, get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpyscf import run_pyscf

geom = [("H", (0, 0, 0)), ("H", (0, 0, BOND))]
moldata = run_pyscf(MolecularData(geom, "sto-3g", 1, 0), run_scf=True, run_fci=False)
qop = jordan_wigner(get_fermion_operator(moldata.get_molecular_hamiltonian()))
n_q = moldata.n_qubits
e_of = sector_ground(get_sparse_operator(qop, n_q).toarray(), n_q, 2)
print(f"openfermion JW ({n_q} qubits): {e_of:.8f} Ha  (diff {abs(e_of - e_fci):.2e})")

assert abs(e_qn - e_fci) < 1e-6 and abs(e_of - e_fci) < 1e-6
print("OK: both routes match PySCF FCI within 1e-6 Ha")
