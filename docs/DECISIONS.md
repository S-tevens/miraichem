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

## 2026-10-02: LiH active space = 2 electrons in 3 orbitals, MOs [1, 2, 5]

- **Decision:** freeze the Li 1s core (MO 0) and keep MOs 1, 2, 5 (the sigma orbitals: bonding 2sigma, 3sigma, 4sigma) with 2 electrons.
  That is 6 qubits with Jordan-Wigner, 4 with Parity (two-qubit reduction).
- **Evidence** (PySCF, STO-3G, total energies in Ha, CASCI vs full FCI):

  | R (A) | HF | CASCI(2e,[1,2,5]) | CASCI(2e,[1,2]) | full FCI |
  |---|---|---|---|---|
  | 1.000 | -7.767362 | -7.782242 | -7.767497 | -7.784460 |
  | 1.595 | -7.862024 | -7.881145 | -7.862286 | -7.882402 |
  | 2.500 | -7.770874 | -7.823077 | -7.773544 | -7.823724 |

- **Alternatives:** (2e, 2 orbitals [1,2]) is only 4 qubits but recovers almost none of the correlation
  (0.3 mHa of ~20 mHa), so VQE would have nothing to find. Adding the pi orbitals [1,2,3,4,5] (10 qubits) gains only ~1 mHa more.
  The default "orbitals centred on the HOMO-LUMO gap" would pick a pi orbital, which cannot mix with the sigma orbitals by symmetry.
- **Caveat:** VQE results for LiH are compared with CASCI in the same active space, not with full FCI. At equilibrium the
  active-space CASCI is 1.3 mHa above full FCI.

## 2026-10-02: Exact-energy helper uses a particle-number penalty

- The qubit Hamiltonian has eigenstates with the wrong electron count. `exact_ground_energy` adds
  `50 * (N - N_target)^2` (built with the same mapper) and verifies the ground state has N_target electrons.
  Works for Jordan-Wigner and Parity without special-casing bit patterns.

## 2026-10-02: Mitigation implemented in-house (readout correction + gate-folding ZNE), not mitiq

- **Decision:** the noisy simulator runs its own measurement pipeline (grouped Pauli measurements, raw counts) and
  mitigation lives in `mitigation/readout.py` and `mitigation/zne.py`. `mitiq` stays installed as a possible cross-check.
- **Why not the ready-made Estimator:** Qiskit's `BackendEstimatorV2` hides the raw counts that readout correction needs.
  Aer's own `EstimatorV2` returns exact expectation values plus Gaussian noise, so it has no real readout error or shot noise.
- **Why not mitiq for ZNE:** mitiq folds with `circuit.inverse()`, which produces gates (`sxdg`) outside the device gate set;
  on Aer those would run noise-free or be re-optimised away, so the noise scaling would be wrong. Our folding repeats only
  two-qubit gates (`ecr`/`cz`/`cx`, all self-inverse: G -> G G G), which dominate error on IBM chips. A "scale factor" is
  therefore the multiple of two-qubit-gate noise. Partial folding gives fractional scales. Only the ansatz is folded, not
  the final basis-change gates.
- **Readout:** per-qubit confusion matrices from two calibration circuits (all |0>, all |1>), inverted on each group's
  outcome distribution (tensored model). Calibration shots are counted in `total_shots`.
- **ZNE:** default scales [1, 3], linear fit; Richardson available. Statistical error follows from the extrapolation weights.
  Variances are taken from the raw distributions (an approximation when readout correction is on).
- **Hardware:** on real devices prefer Runtime's built-in resilience options; the `EnergyBackend` interface is unchanged.
- **Validation:** tests check that the noiseless pipeline equals the exact statevector (X/Y/Z bases), that readout correction
  removes a readout-only noise model, that ZNE moves a device-noise energy toward the ideal value, and that folding preserves the unitary.

## 2026-10-03: LiH verification (2e, 3 orbitals, 1.595 A, STO-3G; error vs CASCI in the same active space)

Sweep: `configs/sweeps/full_lih.yaml` (22 runs, 0 failed, about 6.5 min with 4 workers). Chemical accuracy = error below 1.6 mHa.

**Ideal backend (exact statevector, 300 evaluation budget)**

| Ansatz | Parameters | Result |
|---|---|---|
| UCCSD, reps 1 | 8 (both mappings) | **Reaches chemical accuracy with every optimizer** and both mappings: COBYLA 0.00, L-BFGS-B 0.00, SPSA 0.09 mHa. Parity needs fewer qubits (4 vs 6) and runs about 3x faster. |
| HEA, reps 1 | 24 (JW) / 16 (Parity) | Never reaches chemical accuracy. Best 19.1 mHa (Parity, COBYLA). JW ranges from 20 to 362 mHa depending on the optimizer. |
| HEA, reps 2 | 36 (JW) / 24 (Parity) | Worse than reps 1 at the same budget (35 to 667 mHa): more parameters, same evaluations. |

- L-BFGS-B is the best optimizer for UCCSD (127 evaluations) but gets trapped in local minima for HEA (362 and 583 mHa on JW).
- HEA results depend strongly on the optimizer and budget; none of the 12 ideal HEA runs reached chemical accuracy.
- The HEA starts from the HF state followed by CNOTs, which moves it away from HF (with Jordan-Wigner the CNOT chain
  maps |0101> to |1111> for H2). This is a likely cause of the poor HEA numbers and is worth fixing or documenting
  before presenting HEA as a baseline.

**Noisy simulator (FakeSherbrooke noise, Parity mapping, COBYLA, 1024 shots)**

| Ansatz | Mitigation | Error (mHa) | Two-qubit gates | Transpiled depth |
|---|---|---|---|---|
| UCCSD | none | 556.6 | 203 | 927 |
| UCCSD | readout+zne | 366.5 | 203 | 927 |
| HEA | none | 895.5 | 3 | 19 |
| HEA | readout+zne | 906.8 | 3 | 19 |

- LiH UCCSD is far too deep for current noise levels (203 two-qubit gates): mitigation cuts the error by about 34% but it stays
  hundreds of mHa from chemical accuracy.
- HEA has only 3 two-qubit gates, yet its noisy error (about 900 mHa) is much worse than its ideal error (19 mHa) and
  mitigation does not help. The cause is not yet diagnosed. COBYLA stopped after about 93 evaluations (the ideal run used 301),
  so premature termination under shot noise is one suspect; this needs a follow-up (more shots, different optimizer).

**Feasibility.** Noisy Jordan-Wigner UCCSD (6 qubits, 260 two-qubit gates) takes about 280 s per run and mitigated Parity UCCSD about
180 s, so the noisy part of the LiH sweep is limited to Parity, COBYLA, reps 1. Parity is also faster on the ideal backend.
**Conclusion for the benchmark:** on LiH, UCCSD (with the Parity mapping) is the only configuration that can reach chemical
accuracy, and only on an ideal backend; no tested configuration is close on the noisy simulator.

## 2026-10-03: Extended noisy LiH sweep (resolves the open HEA question)

Sweep: `configs/sweeps/lih_noisy_extended.yaml`, 40 runs (0 failed, about 21.5 min with 4 workers), noisy FakeSherbrooke, Parity
mapping, shots 1024 and 4096, all four mitigation modes, maxiter 300, seed 42 (one seed per configuration).
Best error per group, in mHa (nothing reaches chemical accuracy, 1.6 mHa):

| Ansatz | Optimizer | none | readout | zne | readout+zne | Two-qubit gates |
|---|---|---|---|---|---|---|
| UCCSD (reps 1) | COBYLA | 517.5 | 488.7 | 340.5 | 366.1 | 203 |
| HEA reps 1 | COBYLA | 895.4 | 900.7 | 900.6 | 906.8 | 3 |
| HEA reps 1 | SPSA | 61.3 | **36.6** | 71.3 | 57.9 | 3 |
| HEA reps 2 | COBYLA | 552.5 | 521.7 | 532.8 | 509.0 | 6 |
| HEA reps 2 | SPSA | 119.6 | 341.4 | 105.7 | 653.8 | 6 |

- **The earlier "noisy HEA is stuck near 900 mHa" finding is an optimizer problem, not noise.** COBYLA stops after about 95 evaluations
  (HEA reps 1) and ends at about 900 mHa whatever the mitigation; SPSA with the same ansatz, circuit and noise reaches 37 to 71 mHa using
  its full 301-evaluation budget. COBYLA's trust-region stopping rule fires early on the noisy, shot-limited objective. For noisy runs, SPSA
  is the right default for HEA.
- **UCCSD is limited by circuit depth, not by the optimizer.** 203 two-qubit gates and depth 927 leave it 340 to 517 mHa off. ZNE helps it
  most (517 to 341 mHa, about 34%); readout alone helps little (517 to 489).
- **Mitigation is not uniformly helpful.** For HEA reps 2 with SPSA the spread across modes (106 to 654 mHa) is as large as any mitigation
  effect, so with one seed per configuration these differences are not statistically resolved. The best HEA result (readout, 36.6 mHa) is
  not evidence that readout mitigation beats ZNE for LiH.
- **More shots did not change the picture.** The top three runs mix 1024 and 4096 shots.
- **Conclusion for LiH:** within the tested set no configuration reaches chemical accuracy on the noisy simulator. The shallow HEA with SPSA
  is closest (about 37 to 71 mHa), and UCCSD is accurate only on the ideal backend. Multi-seed replication would be needed before ranking
  the mitigation modes against each other.
- **Caveat:** seed and `maxiter` are fixed in this sweep; shot noise alone can move a result by tens of mHa for SPSA runs.

## 2026-10-18: Seed replication and a statistics-aware recommendation

**Why.** Single-seed results on a noisy objective can be lucky. The best single H2 run of the top configuration was 0.11 mHa while its
mean over 10 seeds was 3.7 mHa, and a configuration that looked best (about 2 mHa) averaged 16 mHa. Ranking by one run per configuration
therefore recommends luck.

**What changed.**
- `seeds` (and `fake_backend_names`) in a sweep YAML repeat every configuration per seed (per device). The seed controls the random start
  point, SPSA's perturbations and shot noise; the config hash includes it, so each replicate is cached separately.
- `miraichem replicate` and `miraichem rank --replicated` group runs that differ only in seed. Per configuration: mean, standard deviation,
  95% t-interval of the mean, median, worst case, and the share of seeds within chemical accuracy with a 95% Wilson interval (well behaved
  for small counts and for 0% / 100%).
- **Recommendation rule** (`benchmark/ranking_replicated.py`):
  1. If some configuration reaches chemical accuracy in at least 50% of its seeds, take the cheapest of those (two-qubit gates, then shots).
  2. Otherwise nothing is reliably accurate: take the lowest-mean configuration, then the cheapest configuration that is **not significantly
     worse** (one-sided Welch t-test, p > 0.05) **and practically close** (mean within 1.5x the best mean plus 1.6 mHa).
- **Why the practical margin.** A very noisy configuration (large variance) can look "not significantly worse" only because the test has no
  power. A unit test with a mean-150 mHa, std-320 mHa configuration confirms the t-test alone would have tied it with a 4 mHa one.
- **Limits.** The 50% reliability level, alpha = 0.05 and the 1.5x margin are judgment calls, exposed as parameters. Ties are decided with
  few seeds (5 to 10), so low power makes ties easy; the justification text states when fewer than 5 seeds are involved. No correction for
  multiple comparisons is applied.
- **Hardware-efficient ansatz start state.** The plain `hea` starts at P|HF> (the CNOT layers permute basis states), not at Hartree-Fock,
  for some qubit counts. The new `hea_hf` prepares the pre-image of HF with X gates, so zero parameters give exactly HF with no extra
  two-qubit gates. It was added as a new ansatz kind rather than changing `hea`, so published results and the hardware runs stay valid.
  With 2 reps on 2 qubits the two CNOT layers cancel and `hea` == `hea_hf` exactly (a built-in consistency check).

### Update 2026-10-18: seed replication of LiH and a cross-device check

- **LiH.** Eight seeds of the HEA configurations and five of UCCSD overturn the single-seed reading above: the 36.6 mHa best run was a lucky
  draw. Averages are 160 to 530 mHa (HEA, SPSA) with seed-to-seed swings of hundreds of mHa and no seed within chemical accuracy; UCCSD is at
  360 mHa on every seed. The statement "the closest is a shallow HEA with SPSA at 36.6 mHa" should be read as "best single run".
  `hea_hf`, which clearly helps H2, does not help LiH (wide intervals; no claim that it is worse).
- **Cross-device.** The same H2 grid on five fake devices (Sherbrooke, Brisbane, Kyiv: ECR; Fez, Torino: CZ) gives rank correlations of
  0.83 to 1.00 between devices and a recommendation that is identical on four of five (and statistically close on the fifth), so the
  ordering of configurations transfers even though absolute errors differ by about 2x between devices. `FakeFez` models `ibm_fez`, the
  device used for the hardware runs.
- **Simulator vs hardware.** Simulated devices underestimate the unmitigated error (hardware 84.5 mHa vs 22 to 61 mHa simulated for UCCSD) but
  agree on the fully mitigated result (3.0 vs 3.3 to 6.6 mHa). Protocols differ (fixed-parameter hardware evaluation vs optimised under noise).


## 2026-10-18: SPSA step-size blow-up at flat starts (new optimizer `spsa_clip`), and what limits LiH

**Finding.** The own SPSA calibrates its step size from the gradient at the start point. Near a stationary point such as the Hartree-Fock
state the gradient is almost zero, the calibrated step becomes huge, and the first updates throw the parameters far away. On LiH the
Hartree-Fock-start HEA (`hea_hf`) started at about 30 mHa and was at 600 to 800 mHa within 30 evaluations; it never recovered
(ideal backend, 8 seeds: 844 mHa mean at 300 evaluations). H2 was unaffected, which is why this went unnoticed. The earlier statement that
`hea_hf` "does not help LiH" was an artifact of this, not a property of the circuit.

**Fix.** A new optimizer kind `spsa_clip` clips every parameter update to +/- 0.05 rad (value chosen on the ideal LiH backend from
0.02 to 0.4; larger values still blow up, smaller ones converge slowly). `spsa` is unchanged, so earlier results stay reproducible; the
two names appear side by side in result tables. Results (8 seeds, mean error): LiH ideal `hea_hf` 844 -> 25 mHa; LiH noisy `hea_hf` +
readout 529 -> 35 mHa (paired p < 0.001); LiH noisy `hea` + readout+ZNE 177 -> 34 mHa (p = 0.09); all H2 configurations unchanged
(for example UCCSD + readout+ZNE 3.7 vs 3.8 mHa).

**What limits LiH once the optimizer is fixed.**
- Hartree-Fock alone is 19.1 mHa off in this active space. The 1-rep HEA, even with no noise at all, ends at about 19 to 35 mHa: it recovers
  essentially none of the correlation energy. More SPSA budget (300 -> 900 -> 1500) or 4x the shots change nothing material.
- Expressivity, not noise, is the first limit. On the ideal backend with exact gradients (L-BFGS-B), the HEA reaches 19 mHa at 1 and 2 reps,
  2.1 mHa (`hea_hf`) at 3 reps and 0.19 mHa at 4 reps (about 9 to 12 two-qubit gates, against 203 for UCCSD).
- Finding that solution with a shot-noise-tolerant optimizer is harder: the clipped SPSA reaches a median of 3.5 to 5 mHa at 3 reps (3000
  evaluations, ideal backend), better than Hartree-Fock but not chemically accurate.
- UCCSD is exact on the ideal backend (0.09 mHa) but its 203 two-qubit gates give a consistent 360 mHa error on the noisy simulator.
- Honest summary: on today's simulated noisy devices, LiH sits between a circuit that is expressive enough but too deep (UCCSD) and
  circuits that are shallow enough but not expressive enough (HEA, 1 to 2 reps).

## 2026-10-19: Error predictor (forecast a noisy run before running it)

**Why.** Benchmarks answer "what worked"; users also need "what will probably work" before spending simulator or hardware
time. The 1,456 noisy runs already saved are exactly the data to learn that from.

**Model.** A random forest (own NumPy implementation, no scikit-learn dependency) on log10(error in mHa). Features are all known
before running: transpiled two-qubit gates and depth, qubits, parameters, shots, budget, mitigation flags, optimizer and circuit
indicators, layers, mapping, the molecule's Hartree-Fock error, and the device's median two-qubit and readout error (medians:
a few very bad qubit pairs made the means misleading, for example 7% for Sherbrooke).

**What was tried.** Baselines: the global mean and the per-molecule mean. Ridge regression with main effects (typical factor
about 2.5x), then with pairwise interactions (about 2.1x), then the forest (1.75x, R2 0.82). The forest won on every held-out device,
so it was kept.

**Validation (all on unseen data).** Five-fold cross-validation grouped by configuration, so no seed of a configuration is in both
training and test. Interval calibration is done fold by fold: the residual quantiles used for fold k come only from the other folds,
and the resulting 80% interval contained the truth 79% of the time. Leave-one-device-out gives 1.7x to 2.4x for four devices.

**Honest limits.**
- Leaving out Sherbrooke gives 6.7x, but Sherbrooke supplies 78% of the data (all of LiH and most H2), so the remaining training set
  is small and narrow. It is reported with a "not a fair test" flag instead of being averaged in or hidden.
- Only two molecules and five devices are in the data. Leave-one-molecule-out was tried and is not reported as a result: with one
  other molecule the problem is nearly degenerate. The predictor warns for anything outside what it saw.
- It forecasts one run with a random seed; the seed-to-seed spread is part of why the interval is wide.
- Features come from the transpile for the device, so a forecast needs that circuit built (a few seconds the first time).

**A bug it exposed.** Saving the model under `results/models/` made the results loader try to parse it as a run and silently return
zero runs (the loader swallows unreadable files so one bad file cannot take the app down). `models` is now a known non-run folder,
with a regression test.

## 2026-10-19: Demo hardening (offline demo, CI, fresh-clone check)

**Offline demo.** A demo that depends on a server, Python and a working install is the first thing to fail in front of judges. The
interface therefore also builds in a read-only mode (`VITE_DEMO=1`) that answers every request from a static snapshot of the
published results (`miraichem export-demo`, 2.3 MB). The snapshot is produced by calling the real API in-process, so its shapes
cannot drift from the live app. The file name for each answer comes from one function, `demo_key`, mirrored in the frontend; both
sides are tested against the same literal examples, so a change on either side fails a test instead of silently breaking the demo.
The demo ignores everything that would need computation (scoring weights, seed-reliability thresholds, forecasts, runs, sweeps,
training) and says so in the interface, instead of showing controls that do nothing.

**CI** runs the Python tests and lint, the frontend typecheck, tests and build, and builds the offline demo as a downloadable
artifact. It does not deploy anything: publishing the demo is a decision for the team.

**A race condition the demo exposed.** The API warms its cache in a background thread at startup, and the first requests could
arrive while it was loading. The second caller saw a cache that looked fresh but was still empty, and the derived data (studies,
devices) then cached that emptiness, so the Overview claimed "no runs" until the files changed. Loading and the derived-data cache are
now guarded by a lock, with a regression test that starts several first requests at once.

