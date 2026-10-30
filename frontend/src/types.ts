// Shapes returned by the MiraiChem API (src/miraichem/api).

export const CHEM_ACC_MHA = 1.6;

export type Backend = "ideal" | "noisy";
export type SourceId = "published" | "local";

export interface Source {
  id: SourceId;
  label: string;
  path: string;
  n_runs: number;
}

export interface RunRow {
  hash: string;
  status: "ok" | "failed";
  molecule: string;
  bond_length: number;
  mapping: string;
  ansatz: string;
  reps: number;
  optimizer: string;
  maxiter: number;
  backend: Backend;
  device: string;
  shots: number;
  mitigation: string;
  seed: number;
  error_mha: number | null;
  chem_acc: boolean | null;
  e_vqe: number | null;
  e_exact: number | null;
  e_hf: number | null;
  hf_error_mha: number | null;
  qubits: number | null;
  params: number | null;
  gates: number | null;
  depth: number | null;
  evals: number | null;
  total_shots: number | null;
  wall_time_s: number | null;
  timestamp: string | null;
  error_message: string | null;
}

export interface RankedRow extends RunRow {
  rank: number;
  score: number;
  cost: number;
  pareto_shots: boolean;
  pareto_gates: boolean;
  label: string;
}

export interface RankResponse {
  source: SourceId;
  bond_length: number;
  bond_lengths: number[];
  ranked: RankedRow[];
  best_hash: string;
  justification: string;
  reached_chemical_accuracy: boolean;
}

export interface ScanRow {
  key: string;
  rank: number;
  label: string;
  hash: string;
  points: number;
  attempted: number;
  worst_error_mha: number;
  mean_error_mha: number;
  coverage: number;
  gates: number;
  shots: number;
  score: number;
}

export interface ScanResponse {
  justification: string;
  best_key: string;
  everywhere: boolean;
  ranked: ScanRow[];
}

export interface RunDetail {
  run: {
    config: Record<string, unknown> & { molecule: { name: string; geometry: string } };
    config_hash: string;
    status: string;
    error_message: string | null;
    convergence_trace: number[];
    convergence_trace_unmitigated: number[];
    optimal_parameters: number[];
    backend_info: Record<string, unknown>;
    job_ids: string[];
    library_versions: Record<string, string>;
    git_commit: string | null;
    timestamp: string | null;
    e_vqe: number | null;
    e_vqe_unmitigated: number | null;
    e_exact: number | null;
    e_hf: number | null;
  };
  row: RunRow;
  command: string;
}

export interface Config {
  label: string;
  mapping: string;
  ansatz: string;
  reps: number;
  optimizer: string;
  maxiter: number;
  shots: number;
  mitigation: string;
  device: string;
  bond_length: number;
  errors: number[];
  n_seeds: number;
  mean: number;
  std: number;
  ci95: number;
  median: number;
  worst: number;
  best: number;
  n_chem: number;
  p_chem: number;
  p_chem_lo: number;
  p_chem_hi: number;
  two_qubit_gates: number;
  total_shots: number;
  rank: number;
}

export interface Recommendation {
  label: string;
  reliable: boolean;
  justification: string;
  lowest_mean_label: string;
  tied_labels: string[];
  params: { reliability: number; alpha: number; margin_factor: number };
}

export interface Study {
  id: string;
  molecule: string;
  backend: Backend;
  device: string;
  bond_length: number;
  hf_error_mha: number | null;
  configs: Config[];
  recommendation: Recommendation | null;
}

export interface DeviceBlock {
  molecule: string;
  devices: string[];
  configs: { name: string; means: Record<string, number | null> }[];
  spearman: Record<string, Record<string, number | null>>;
  recommendations: Record<
    string,
    { label: string; mean: number; std: number; n_chem: number; n_seeds: number }
  >;
  summary: string;
}

export interface HardwareEvaluation {
  mitigation: string;
  error_mha: number;
  std_mha: number;
  job_id: string;
}

export interface HardwareResult {
  source: string;
  source_label: string;
  molecule: string;
  bond_length: number;
  device: string;
  shots: number;
  timestamp: string;
  hf_error_mha: number;
  ideal_error_mha: number;
  simulator_error_mha: number;
  simulator_mitigation: string;
  evaluations: HardwareEvaluation[];
}

export interface Curve {
  molecule: string;
  label: string;
  backend: string;
  bond_lengths: number[];
  e_hf: number[];
  e_exact: number[];
  e_vqe: (number | null)[];
  error_mha: (number | null)[];
  hf_error_mha: number[];
}

export interface Overview {
  source: SourceId;
  n_runs: number;
  n_ok: number;
  n_failed: number;
  by_molecule: Record<string, Record<string, number>>;
  hardware_results: number;
  curves: number;
  has_replication: boolean;
  best_noisy: RunRow | null;
  hf_error_mha: Record<string, number>;
  recent: RunRow[];
}

export interface Choice {
  id: string;
  label: string;
  hint?: string;
}

export interface MoleculeInfo {
  id: string;
  name: string;
  geometry: string;
  basis: string;
  active_electrons: number | null;
  active_orbitals: number | number[] | null;
  equilibrium: number;
  bond_lengths: number[];
}

export interface Options {
  mappings: Choice[];
  ansatzes: Choice[];
  optimizers: Choice[];
  backends: Choice[];
  mitigations: Choice[];
  devices: Choice[];
  molecules: MoleculeInfo[];
}

export interface RunRequest {
  molecule: string;
  bond_length: number;
  mapping: string;
  ansatz: string;
  ansatz_reps: number;
  optimizer: string;
  maxiter: number;
  backend: Backend;
  fake_backend_name: string;
  shots: number;
  mitigation: string;
  seed: number;
  force: boolean;
}

export interface Job {
  id: string;
  kind: "run" | "sweep" | "train" | "sqd" | "sqd-plan" | "sqd-hardware";
  title: string;
  status: "queued" | "running" | "done" | "failed";
  created: number;
  started: number | null;
  finished: number | null;
  total: number;
  completed: number;
  cached: number;
  failed_runs: number;
  message: string;
  error: string | null;
  result_hashes: string[];
  best_error_mha: number | null;
  details: Record<string, string>;
  elapsed_s: number;
}

export interface SweepInfo {
  name: string;
  description?: string;
  molecule?: string;
  n_runs?: number;
  n_cached?: number;
  backends?: string[];
  ansatzes?: string[];
  optimizers?: string[];
  seeds?: number;
  devices?: string[];
  error?: string;
}

export interface Reference {
  e_hf: number;
  e_exact: number;
  hf_error_mha: number;
  exact_method: string;
  n_qubits_jw: number;
}

export interface PredictionMetrics {
  mae_log10: number;
  rmse_log10: number;
  typical_factor: number;
  r2: number;
  n: number;
}

export interface PredictorReport {
  trained: string;
  n_runs: number;
  n_configurations: number;
  molecules: string[];
  devices: string[];
  cross_validation: PredictionMetrics;
  baseline_mean: PredictionMetrics;
  baseline_per_molecule: PredictionMetrics;
  per_molecule: Record<string, PredictionMetrics>;
  interval: { level: number; coverage: number; median_width_factor: number };
  device_holdout: (PredictionMetrics & { device: string; train_rows: number; fair: boolean })[];
  importance: Record<string, number>;
  points: { predicted: number; actual: number; molecule: string; device: string }[];
}

export interface PredictorStatus {
  available: boolean;
  origin?: "local" | "published";
  report?: PredictorReport;
  noisy_runs_available: number;
  newer_data?: boolean;
}

export interface PredictionResult {
  supported: boolean;
  reason?: string;
  origin?: string;
  prediction?: {
    error_mha: number;
    low_mha: number;
    high_mha: number;
    p_chemical_accuracy: number;
    p_beats_hartree_fock: number;
    hf_error_mha: number;
    warnings: string[];
  };
  circuit?: { two_qubit_gates: number; depth: number; qubits: number; parameters: number };
  similar?: RunRow[];
}

// ---------------------------------------------------------------- SQD

export interface SqdHardwareInfo {
  backend: string;
  job_id: string;
  circuit: string;
  shots?: number;
  estimated_qpu_s?: number;
  qpu_seconds: number | null;
  two_qubit_gates?: number;
  depth?: number;
  qubits_used?: number;
  duration_s?: number;
  ideal_same_circuit: {
    error_mha: number | null;
    below_hf_mha: number | null;
    distinct_samples: number;
  } | null;
}

export interface SqdRow {
  hash: string;
  status: "ok" | "failed";
  error_message: string | null;
  molecule: string;
  bond_length: number;
  n_reps: number;
  shots: number;
  readout_noise: number;
  noise_model: string | null;
  sampler: SqdSampler;
  noise_summary: { backend: string; survival: number; two_qubit_gates: number; median_readout_error: number } | null;
  spin: number;
  iterations: number;
  samples_per_batch: number;
  num_batches: number;
  seed: number;
  qubits: number | null;
  e_sqd: number | null;
  e_hf: number | null;
  e_ccsd: number | null;
  e_exact: number | null;
  error_mha: number | null;
  hf_error_mha: number | null;
  ccsd_error_mha: number | null;
  below_hf_mha: number | null;
  below_ccsd_mha: number | null;
  energy_per_iteration: number[];
  hardware: SqdHardwareInfo | null;
  chem_acc: boolean | null;
  n_determinants: number | null;
  subspace_dim: number | null;
  subspace_fraction: number | null;
  distinct_samples: number | null;
  postselected: number | null;
  error_per_iteration_mha: number[];
  wall_time_s: number | null;
  sampling_time_s: number | null;
  diagonalisation_time_s: number | null;
  fci_time_s: number | null;
  ccsd_time_s: number | null;
  timestamp: string | null;
}

export interface SqdGroup {
  molecule: string;
  bond_length: number;
  n_reps: number;
  shots: number;
  readout_noise: number;
  noise_model: string | null;
  sampler: SqdSampler;
  iterations: number;
  samples_per_batch: number;
  num_batches: number;
  qubits: number | null;
  n_seeds: number;
  mean_error_mha: number;
  std_error_mha: number;
  best_error_mha: number;
  worst_error_mha: number;
  median_error_mha: number;
  share_chem_acc: number;
  hf_error_mha: number | null;
  ccsd_error_mha: number | null;
  mean_subspace_fraction: number;
  mean_wall_time_s: number;
  errors: number[];
  seeds: number[];
}

/** Where an SQD run's bitstrings came from: the circuit, or one of two controls that leave it out. */
export type SqdSampler = "circuit" | "hf_noise" | "uniform";

/** Error of the subspace controls at one subspace size (strings per spin), in mHa against exact. */
export interface SqdControlPoint {
  k: number;
  dim: number;
  random_mean_mha: number;
  random_std_mha: number;
  excitation_mha: number;
  oracle_mha: number;
}

export interface SqdControl {
  molecule: string;
  qubits: number;
  bond_length: number;
  n_strings: number;
  e_exact: number;
  hf_error_mha: number;
  ccsd_error_mha: number;
  n_random: number;
  curve: SqdControlPoint[];
  sqd: (SqdControlPoint & { seed: number; n_reps: number; error_mha: number })[];
}

export interface SqdResponse {
  source: SourceId;
  runs: SqdRow[];
  groups: SqdGroup[];
  /** Each molecule's equilibrium bond length; runs at any other distance belong to the stretched study. */
  default_bonds: Record<string, number>;
  /** Subspace-control studies, one per molecule; empty when none were run. */
  controls: SqdControl[];
}

export interface SqdOptions {
  molecules: {
    id: string;
    name: string;
    basis: string;
    spin: number;
    default_bond_length: number;
    description: string;
    qubits: number;
    determinants: number;
    simulable: boolean;
    has_exact: boolean;
    exact_cached: boolean;
    exact_estimate_s: number;
  }[];
}

export interface StructureOrbital {
  index: number;
  energy: number;
  occupation: number;
  tag: "HOMO" | "LUMO" | "";
  pos: { v: number[]; f: number[] };
  neg: { v: number[]; f: number[] };
}

export interface Structure {
  id: string;
  label: string;
  kind: "computed" | "schematic";
  note: string;
  atoms: [string, number, number, number][];
  orbitals: StructureOrbital[];
}

export interface SqdRunRequest {
  molecule: string;
  bond_length: number | null;
  n_reps: number;
  shots: number;
  readout_noise: number;
  noise_model: string | null;
  exact: boolean;
  iterations: number;
  samples_per_batch: number;
  num_batches: number;
  seed: number;
}

export interface SqdHardwareStatus {
  enabled: boolean;
  token_configured: boolean;
  max_qpu_seconds: number;
  confirm_phrase: string;
  plan_valid_seconds: number;
}

export interface SqdHardwarePlan {
  plan_id: string;
  expires_at: number;
  molecule: string;
  backend: string;
  qubits: number;
  qubits_used: number;
  two_qubit_gates: number;
  depth: number;
  shots: number;
  per_shot_us: number;
  estimated_qpu_s: number;
  estimated_post_s: number;
  remaining_qpu_s: number | null;
  budget_s: number;
  cap_s: number;
  pending_jobs: number | null;
  exact_available: boolean | null;
  n_determinants: number | null;
  problems: string[];
  confirm_phrase: string;
  expired: boolean;
  submitted: boolean;
}

// ---- trust report

export type TrustStatus = "pass" | "warn" | "fail" | "none";
export type TrustVerdict = "trusted" | "partly" | "not_yet" | "no_evidence";

export interface TrustItem {
  id: string;
  title: string;
  status: TrustStatus;
  detail: string;
  route: string | null;
}

export interface TrustReport {
  kind: "vqe" | "sqd";
  subject: { molecule: string; backend?: string; device?: string; bond_length: number | null; label: string | null };
  verdict: TrustVerdict;
  verdict_text: string;
  sentence: string;
  evidence: TrustItem[];
  rules: string[];
}

export interface TrustSubject {
  kind: "vqe" | "sqd";
  molecule: string;
  backend: string;
  device: string;
  bond_length?: number;
  label?: string;
}
