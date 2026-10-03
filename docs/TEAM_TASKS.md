# Team task split

Four owners. Everyone commits their own work under their own name, opens a PR into `main`, and
gets one review from a teammate. Every number in docs, notebooks and the dashboard must come from
code we actually ran. Each owner must be able to explain their modules and the science behind them
to the judges.

Deadline: Round 1 closes **7 Oct 2026, 11:59 PM IST**. Target: feature-complete by the evening of
**5 Oct**; 6-7 Oct are for docs, video and submission buffer.

Updated 3 Oct: work rebalanced (see "What is left" per person). Swap areas among yourselves if you
prefer, then update this file.

## Shared contracts

- `config.py` (pydantic): `MoleculeConfig`, `RunConfig`, `SweepConfig`, `RunConfig.config_hash()`.
- `QubitProblem`: `hamiltonian` (SparsePauliOp), `energy_shift`, `num_qubits`, `num_particles`,
  `num_spatial_orbitals`, `mapping`, `mapper`.
- Backend interface (`backends/base.py`): `estimate(circuit, observable, parameter_values) -> EstimateResult`.
- `RunResult` (`vqe/result.py`): one JSON file per run in `results/<molecule>/<hash>.json`.
- All energies are **total energies in Hartree**; geometry in Angstrom.

## @S-tevens: Chemistry, ranking, curves, dashboard

Files: `config.py`, `chemistry/`, `mapping/`, `configs/molecules/`, `benchmark/ranking.py`,
`analysis/dissociation.py`, `dashboard/`.

In `main`: PySCF references (HF, FCI/CASCI), qubit Hamiltonians (Jordan-Wigner, Parity), LiH active
space (rationale in `docs/DECISIONS.md`), config models and hashing, with tests.

What is left:
- **Ranking** (`benchmark/ranking.py`): weighted score with configurable weights, Pareto fronts over
  (error, total shots) and (error, two-qubit gates), `recommend()` with a justification string built
  from real numbers, and the CLI `rank` command. Tests on synthetic data (`test_ranking.py`).
- **Dissociation curve** (`analysis/dissociation.py`): HF vs VQE (best config) vs exact over 10-15
  bond lengths for H2, with a VQE-error panel and the chemical-accuracy band shaded.
- **Streamlit dashboard** (`dashboard/app.py`): reads only `results/`; Overview, Leaderboard,
  Convergence, Dissociation curve, Mitigation (before vs after), Recommendation tabs.

Done when: dashboard renders from saved results end to end.

## @jeev-69: Ansatz, optimizers, LiH, slides

Files: `ansatz/`, `optim/`, `vqe/runner.py`, `backends/ideal.py`.

In `main`: HEA and UCCSD from the Hartree-Fock state, COBYLA / SPSA / L-BFGS-B behind one interface
(`maxiter` = evaluation budget), ideal statevector backend, `run_vqe` with per-run failure capture,
`miraichem run`; tests in `test_vqe_ideal.py` and `test_optimizers.py`.

What is left:
- **LiH verification**: run LiH with both mappings and both ansatzes on ideal and noisy; tune
  `maxiter` and report which configurations reach chemical accuracy; `configs/sweeps/full_lih.yaml`.
- **Optimizer and ansatz comparison**: tune SPSA (matched evaluation budgets), check the HEA
  initialisation, write the findings into `docs/DECISIONS.md`.
- **Slides (Canva template)**: problem, significance and proposed-solution slides; draft the video script.

Done when: LiH results exist for both mappings and the findings are written up.

## @ritesh9116: Noisy backend, mitigation, hardware

Files: `backends/noisy.py`, `backends/hardware.py`, `mitigation/`.

In `main`: Aer noise model and coupling map from a fake IBM backend, shot-based measurement with raw
counts, transpiled depth and two-qubit gate counts, readout mitigation and gate-folding ZNE
(rationale in `docs/DECISIONS.md`), mitigated and unmitigated values recorded per evaluation,
`scripts/ibm_check.py`; tests in `test_vqe_noisy.py` and `test_mitigation.py`.

What is left:
- **Hardware backend** (`backends/hardware.py`): disabled unless `--allow-hardware`; prints the
  number of circuits, shots and expected QPU time; requires interactive `y/N` confirmation; picks
  the least-busy backend unless one is named; saves job IDs so results can be re-fetched; default
  mode is fixed-parameter evaluation (take optimal parameters from the noisy simulator, evaluate
  on hardware). Runtime's built-in resilience options on hardware.
- **CLI `hardware` command** and a dry run that prints its estimate and aborts safely.
- **Slides**: quantum-relevance and hardware slides.

Budget: the free plan allows 600 s of QPU time per period. Treat it like money; always confirm.

Done when: a hardware dry run prints its estimate and aborts safely, and one real H2 evaluation is saved.

## @rohithrajha: Sweeps, CLI, notebook, documentation

Files: `benchmark/sweep.py`, `benchmark/metrics.py`, `benchmark/storage.py`, `cli.py`,
`configs/sweeps/`, `notebooks/`, `docs/`, `README.md`.

In `main`: `SweepConfig.expand()` (Cartesian product with exclusion rules), `metrics.py`,
`storage.py`, a first `cli.py` (`run`, `reference`).

What is left:
- **Sweep engine** (`benchmark/sweep.py`): run an expanded sweep, skip results already cached by
  hash unless `--force`, `--max-workers` for parallel simulator runs, progress bar, failed runs do
  not stop the sweep. CLI `sweep` command. Configs `quick_h2.yaml` (under 2 min, used by the demo)
  and `full_h2.yaml`. Tests in `test_sweep.py`, `test_metrics.py`.
- **Summary export**: load all results into a pandas DataFrame, Parquet/CSV summary; CLI `report`
  writing `docs/RESULTS.md` from real saved results.
- **Documentation**: README (all sections the hackathon requires), `ARCHITECTURE.md` with a
  mermaid diagram, `SCIENCE_PRIMER.md`, setup steps (Linux/WSL2 required, see `docs/DECISIONS.md`).
- **Demo notebook** `01_h2_end_to_end_demo.ipynb`: runs top to bottom in a few minutes; calls the package only.

Done when: `miraichem sweep configs/sweeps/quick_h2.yaml` produces saved results and the notebook runs on a fresh clone.

## Timeline

| Day | Goal | Owners |
|-----|------|--------|
| Fri 2 Oct | Environment, compatibility checks, H2 spike, chemistry and Hamiltonian tests, ideal and noisy VQE, mitigation | in `main` |
| Sat 3 Oct | Sweep engine and `quick_h2`. Ranking on synthetic data. Hardware backend with dry run. LiH verification. | @rohithrajha: sweep. @S-tevens: ranking. @ritesh9116: hardware. @jeev-69: LiH |
| Sun 4 Oct | `full_h2` and `full_lih` sweeps run. Leaderboard and recommendation from real results. Submit hardware jobs early (queue times). Dissociation curve. | @rohithrajha: sweeps. @S-tevens: recommend, curve. @ritesh9116: hardware jobs. @jeev-69: LiH write-up |
| Mon 5 Oct | Dashboard complete. Demo notebook. Hardware results collected. Slides drafted. | @S-tevens: dashboard. @rohithrajha: notebook, README. @ritesh9116: hardware results. @jeev-69: slides |
| Tue 6 Oct | RESULTS.md from real data. Fresh-clone test. Figures. Slides finished, demo video recorded (every member's own voice). | Everyone |
| Wed 7 Oct | Buffer only: re-record if needed, final push, submit well before 11:59 PM IST. | Everyone |

Risks: submit hardware jobs by 4 Oct so 5 Oct only collects results. If LiH takes too long, ship H2
fully and present LiH as an extra. Video narration must be recorded by team members (hackathon rule).

## Workflow

1. `git checkout -b feat/<area>-<topic>`
2. Small commits using conventional messages (`feat:`, `fix:`, `test:`, `docs:`).
3. `ruff check . && ruff format . && pytest -m "not slow"` before pushing.
4. Open a PR, get one review, merge.
5. Everyone must be able to explain their module and its science to the judges.
