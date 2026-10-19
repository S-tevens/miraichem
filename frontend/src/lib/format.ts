/** Anything with the configuration fields (a replicated Config or a single RunRow). */
export interface CfgLike {
  ansatz: string;
  reps: number;
  optimizer: string;
  mitigation: string;
  mapping: string;
  maxiter: number;
  shots: number;
}

/** Errors in milli-Hartree, with just enough digits to be honest about precision. */
export function mha(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  const a = Math.abs(value);
  if (a >= 100) return value.toFixed(0);
  if (a >= 10) return value.toFixed(1);
  if (a >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

export const pct = (x: number): string => `${Math.round(x * 100)}%`;

export const ANSATZ: Record<string, string> = {
  uccsd: "UCCSD",
  hea: "Hardware-efficient",
  hea_hf: "Hardware-efficient, HF start",
};
export const OPTIMIZER: Record<string, string> = {
  cobyla: "COBYLA",
  spsa: "SPSA",
  spsa_clip: "SPSA, clipped",
  lbfgsb: "L-BFGS-B",
};
export const MITIGATION: Record<string, string> = {
  none: "No mitigation",
  readout: "Readout",
  zne: "ZNE",
  "readout+zne": "Readout + ZNE",
};
export const MAPPING: Record<string, string> = {
  parity: "Parity",
  jordan_wigner: "Jordan-Wigner",
};

export interface LabelParts {
  ansatz: string;
  optimizer: string;
  mitigation: string;
  mapping: string;
  budget: string;
}

export function parts(c: CfgLike): LabelParts {
  const depth = c.ansatz === "uccsd" ? "" : `, ${c.reps} ${c.reps === 1 ? "layer" : "layers"}`;
  return {
    ansatz: `${ANSATZ[c.ansatz] ?? c.ansatz}${depth}`,
    optimizer: OPTIMIZER[c.optimizer] ?? c.optimizer,
    mitigation: MITIGATION[c.mitigation] ?? c.mitigation,
    mapping: MAPPING[c.mapping] ?? c.mapping,
    budget: `${c.maxiter} evaluations${c.shots ? `, ${c.shots.toLocaleString("en")} shots` : ""}`,
  };
}

export function shortName(c: CfgLike): string {
  const p = parts(c);
  return `${p.ansatz} · ${p.optimizer} · ${p.mitigation}`;
}

export const deviceName = (d: string): string => d.replace(/^Fake/, "");

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "n/a";
  if (seconds < 1) return "<1 s";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const m = Math.floor(seconds / 60);
  if (m < 90) return `${m} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en", { dateStyle: "medium", timeStyle: "short" });
}

export const moleculeName = (m: string): string => (m === "H2" ? "H₂" : m);
