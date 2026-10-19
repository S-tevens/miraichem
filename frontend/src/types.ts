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
  kind: "run" | "sweep" | "train";
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
