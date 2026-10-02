# Decisions log

Format: date, decision, alternatives considered, reason.

## 2026-10-02: Hamiltonian stack = qiskit-nature (OpenFermion kept as fallback)

- **Decision:** use `qiskit-nature 0.8.0` (`PySCFDriver`, `ActiveSpaceTransformer`, mappers, `UCCSD`, `HartreeFock`).
- **Evidence:** it installs next to `qiskit 2.5.2` (its requirement is `qiskit>=1.4,<3.0`, plus `qiskit-algorithms>=0.4.0`).
  `scripts/spike_h2.py` builds the H2/STO-3G Jordan-Wigner Hamiltonian with it. Its exact energy in the
  2-electron sector matches PySCF FCI to 4e-16 Ha. UCCSD for H2 builds with 3 parameters.
- **Alternative:** `openfermion 1.8.1` + `openfermionpyscf 0.5`. Also works (matches FCI to 0.0), kept as a fallback
  if qiskit-nature breaks on a future Qiskit release. It would need our own `SparsePauliOp` adapter.
- **Reason:** less code for us and a ready UCCSD / HF state, so we spend time on the benchmarking, not on plumbing.

## 2026-10-02: Python 3.11 on Linux/WSL2; native Windows is not supported

- **Decision:** develop and run in Python 3.11 on Linux (WSL2 Ubuntu on Windows, or macOS/Linux directly).
- **Problem:** `pyscf` publishes no Windows wheels, and building from source on Windows fails (needs cmake and a C++ toolchain).
- **Alternatives:** conda-forge pyscf on Windows (extra tool, untested); Docker (heavier). WSL2 was the quickest and works.
- **Note:** everything else (qiskit, aer, runtime, mitiq, streamlit) installs on native Windows. Only chemistry needs Linux.
- **Setup:** `scripts/wsl_setup.sh`. Keep the venv in the Linux home dir (`~/mc-venv`), not under `/mnt/c`, for speed.

## 2026-10-02: Package manager = uv

- Fast, resolves Python 3.11 automatically, same commands on Windows, WSL and Linux.

## 2026-10-02: Library versions confirmed together

See `requirements.txt` for the full pin list. Key versions: qiskit 2.5.2, qiskit-aer 0.17.2,
qiskit-ibm-runtime 0.50.0, qiskit-nature 0.8.0, qiskit-algorithms 0.4.0, pyscf 2.14.0, mitiq 1.1.0,
openfermion 1.8.1, numpy 2.2.6, scipy 1.17.1.

- Fake backends import from `qiskit_ibm_runtime.fake_provider` (e.g. `FakeSherbrooke`, 127 qubits); noise model builds via
  `qiskit_aer.noise.NoiseModel.from_backend`.
- `qiskit_aer.primitives.EstimatorV2` is available. Hardware primitives are Runtime V2 (`EstimatorV2`).
- Hardware-efficient ansatz: `qiskit.circuit.library.efficient_su2` (function form in Qiskit 2.x).

## Open (decide in Phase 3)

- ZNE on Aer: `mitiq` (installs fine) vs Runtime resilience options. To be tested by the mitigation owner.
- IBM account allowance: needs the team's API token; not yet checked.
