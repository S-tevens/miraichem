# Team task split

Four owners, one per area. Everyone commits their own work under their own name, opens a PR into
`main`, and gets one review from a teammate. Every number in docs, notebooks and the dashboard
must come from code we actually ran.

Deadline: Round 1 closes **7 Oct 2026, 11:59 PM IST**. Aim to be feature-complete by the evening of 6 Oct.

Assignments below are a first proposal; swap areas among yourselves if you prefer, then update this file.

## Shared contracts (agree on these first, day 1)

- `config.py` (pydantic): `MoleculeConfig`, `RunConfig`, `SweepConfig`, `RunConfig.config_hash()`.
- `QubitProblem` dataclass: `hamiltonian` (SparsePauliOp), `energy_shift`, `num_qubits`,
  `num_particles`, `num_spatial_orbitals`.
- Backend interface: `estimate(circuit, observable, parameter_values) -> (value, std_error, metadata)`.
- `RunResult` dataclass (fields in the project brief, section 6.8).
- All energies are **total energies in Hartree**; geometry in Angstrom.

Owner of the shared contracts: Member A. Others code against them.

## Member A: Chemistry and Hamiltonian (`@S-tevens`)

Files: `config.py`, `chemistry/`, `mapping/`, `configs/molecules/`.

- Build PySCF molecules; HF and FCI/CASCI reference energies (`ClassicalReference`).
- LiH active space (4-6 qubits), justified in `docs/DECISIONS.md`.
- `build_qubit_hamiltonian` for Jordan-Wigner and Parity; exact-diagonalisation helper.
- Tests: `test_classical.py`, `test_hamiltonian.py` (qubit H + shift matches PySCF within 1e-6 Ha,
  H2 and LiH, both mappings), qubit counts, config hashing.

Done when: those tests pass.

## Member B: Ansatz, optimizers, VQE loop (`@jeev-69`)

Files: `ansatz/`, `optim/`, `vqe/runner.py`, `backends/ideal.py`.

- HEA and UCCSD, both starting from the Hartree-Fock state.
- Uniform `minimize(fun, x0, maxiter, callback) -> OptResult` over COBYLA, SPSA (seeded), L-BFGS-B.
- Ideal statevector backend; `run_vqe(run_cfg) -> RunResult` with convergence trace; per-run failure capture.
- Tests: HF-state energy equals PySCF HF within 1e-6 Ha; ideal UCCSD + L-BFGS-B reaches H2 exact
  energy within 1.6 mHa (`test_vqe_ideal.py`).

Done when: `miraichem run ... --backend ideal` produces a correct H2 result JSON.

## Member C: Noisy backends, mitigation, hardware (`@ritesh9116`)

Files: `backends/noisy.py`, `backends/hardware.py`, `mitigation/`.

- Aer noise model and coupling map from a fake IBM backend; transpiled depth and 2-qubit gate counts.
- Readout mitigation and ZNE (decision recorded in `docs/DECISIONS.md`), recording mitigated and
  unmitigated values.
- Hardware backend: disabled unless `--allow-hardware`, prints cost estimate, requires `y/N`
  confirmation, saves job IDs; default mode is fixed-parameter evaluation.
- Tests: noisy-simulator determinism with a fixed seed.

Done when: a noisy H2 run is reproducible and a hardware dry run prints its estimate and aborts safely.

## Member D: Benchmark, dashboard, docs (`@rohithrajha`)

Files: `benchmark/`, `analysis/`, `cli.py`, `dashboard/`, `notebooks/`, `docs/`, `README.md`.

- Sweep expansion with exclusion rules and caching by hash; storage (JSON per run, DataFrame export).
- Metrics, weighted ranking, Pareto fronts, `recommend()`.
- Dissociation-curve analysis; Streamlit dashboard reading only `results/`.
- Typer CLI, README, ARCHITECTURE.md (mermaid), SCIENCE_PRIMER.md, RESULTS.md, demo notebook.
- Tests: `test_sweep.py`, `test_metrics.py`, `test_ranking.py` on synthetic data.

Done when: dashboard renders from saved results end to end.

## Timeline

| Date | Goal |
|------|------|
| 2 Oct | Repo, environment, Hamiltonian spike; contracts agreed |
| 3 Oct | Chemistry and Hamiltonian tests green; ansatz and optimizers started |
| 4 Oct | Ideal and noisy VQE working; metrics and ranking on synthetic data |
| 5 Oct | Mitigation and sweeps; hardware gate in place; dashboard skeleton |
| 6 Oct | Hardware run (with team confirmation), curves, dashboard complete |
| 7 Oct | Polish, video, fresh-clone test, submission |

## Workflow

1. `git checkout -b feat/<area>-<topic>`
2. Small commits using conventional messages (`feat:`, `fix:`, `test:`, `docs:`).
3. `ruff check . && ruff format . && pytest -m "not slow"` before pushing.
4. Open a PR, get one review, merge.
5. Everyone must be able to explain their module and its science to the judges.
