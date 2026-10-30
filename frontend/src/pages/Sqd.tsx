import { useEffect, useMemo, useState } from "react";
import { ApiError, api, qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import { DEMO } from "../lib/demo";
import { mha } from "../lib/format";
import { log, logTicks } from "../lib/scale";
import { useWidth } from "../lib/hooks";
import { SqdHardware } from "../components/SqdHardware";
import { MoleculeViewer } from "../components/MoleculeViewer";
import { SqdScaling } from "../components/SqdScaling";
import { duration } from "../lib/format";
import { Empty, Field, Loading, Notice, PageHead, Panel, Select, Stat } from "../ui/ui";
import {
  CHEM_ACC_MHA,
  type Job,
  type SqdGroup,
  type SqdOptions,
  type SqdResponse,
  type SqdRow,
  type SqdRunRequest,
} from "../types";
import "../components/charts.css";
import "./NewRun.css";
import "./Sqd.css";

const SUB = "₀₁₂₃₄₅₆₇₈₉";
/** "N2" -> "N₂"; "N2_631G" -> "N₂ (631G)" (the part after the underscore is the basis set). */
const molName = (n: string) => {
  const [head, ...rest] = n.split("_");
  const f = head.replace(/\d/g, (d) => SUB[Number(d)]);
  return rest.length ? `${f} (${rest.join(" ")})` : f;
};
const secs = (s: number | null | undefined) =>
  s === null || s === undefined ? "n/a" : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
const ha = (e: number | null | undefined) => (e === null || e === undefined ? "n/a" : e.toFixed(6));
const int = (n: number | null | undefined) => (n === null || n === undefined ? "n/a" : n.toLocaleString("en"));

function groupLabel(g: SqdGroup): string {
  const parts = [g.noise_model ? `${g.noise_model} noise model` : `noise ${g.readout_noise}`];
  if (g.samples_per_batch !== 300) parts.push(`${g.samples_per_batch} samples per batch`);
  if (g.iterations !== 3) parts.push(`${g.iterations} ${g.iterations === 1 ? "round" : "rounds"}`);
  if (g.n_reps !== 1) parts.push(`${g.n_reps} circuit layers`);
  return parts.join(", ");
}

function sameGroup(a: SqdRow, g: SqdGroup): boolean {
  return (
    a.molecule === g.molecule &&
    a.bond_length === g.bond_length &&
    a.n_reps === g.n_reps &&
    a.shots === g.shots &&
    a.readout_noise === g.readout_noise &&
    a.noise_model === g.noise_model &&
    a.iterations === g.iterations &&
    a.samples_per_batch === g.samples_per_batch &&
    a.num_batches === g.num_batches
  );
}

/**
 * A typical run: the median seed of the default setting (the one the command line and the form start
 * with), so the page never opens on the best setting or a lucky seed.
 */
function typicalRun(rows: SqdRow[], groups: SqdGroup[]): SqdRow | undefined {
  const ok = rows.filter((r) => r.status === "ok");
  const simulated = ok.filter((r) => !r.hardware && r.error_mha !== null);
  const isDefault = (g: SqdGroup) =>
    g.n_reps === DEFAULTS.n_reps &&
    g.shots === DEFAULTS.shots &&
    g.readout_noise === DEFAULTS.readout_noise &&
    g.noise_model === null &&
    g.iterations === DEFAULTS.iterations &&
    g.samples_per_batch === DEFAULTS.samples_per_batch &&
    g.num_batches === DEFAULTS.num_batches;
  const g =
    [...groups].filter(isDefault).sort((a, b) => b.n_seeds - a.n_seeds)[0] ??
    [...groups].sort((a, b) => b.n_seeds - a.n_seeds || a.mean_error_mha - b.mean_error_mha)[0];
  if (!g) return simulated[0] ?? ok[0];
  const members = simulated.filter((r) => sameGroup(r, g));
  return members.sort((a, b) => Math.abs(a.error_mha! - g.median_error_mha) - Math.abs(b.error_mha! - g.median_error_mha))[0] ?? simulated[0] ?? ok[0];
}

// ------------------------------------------------------------------ log-scale rows

interface LogRow {
  key: string;
  label: string;
  sub?: string;
  value: number;
  tone: "hf" | "cc" | "q" | "full";
  text?: string;
}

function LogRows({
  rows,
  domain,
  band,
  axis,
  bars,
  aria,
}: {
  rows: LogRow[];
  domain: [number, number];
  band?: boolean;
  axis: string;
  bars?: boolean;
  aria: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const compact = width < 560;
  const left = compact ? 14 : 232;
  const right = 36;
  const rowH = compact ? 70 : 46;
  const top = 26;
  const x = log(domain, [left, Math.max(left + 140, width - right)]);
  const height = top + rows.length * rowH + 40;
  const bottom = top - 8 + rows.length * rowH;
  const ticks = logTicks(domain).filter((t) => Math.log10(t) % 1 === 0);
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={height} role="img" aria-label={aria}>
        {band && <rect x={x(domain[0])} y={top - 8} width={x(CHEM_ACC_MHA) - x(domain[0])} height={rows.length * rowH} className="chart__band" />}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 8} y2={bottom} className="chart__grid" />
            <text x={x(t)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{t >= 1000 ? `${t / 1000}k` : t}</text>
          </g>
        ))}
        {rows.map((r, k) => {
          const cy = top + k * rowH + (compact ? rowH - 20 : rowH / 2 - 6);
          const cls = r.tone;
          return (
            <g key={r.key}>
              <text x={compact ? left : left - 14} y={compact ? cy - 28 : cy + 4} textAnchor={compact ? "start" : "end"} className="sqd__row">
                {r.label}
              </text>
              {r.sub && (
                <text x={compact ? left : left - 14} y={compact ? cy - 12 : cy + 20} textAnchor={compact ? "start" : "end"} className="sqd__label">
                  {r.sub}
                </text>
              )}
              <line x1={x(domain[0])} x2={x(domain[1])} y1={cy} y2={cy} className="sqd__track" />
              {bars ? (
                <rect x={x(domain[0])} y={cy - 8} width={Math.max(2, x(r.value) - x(domain[0]))} height={16} rx={3} className={`sqd__bar sqd__bar--${cls}`} />
              ) : (
                <circle cx={x(r.value)} cy={cy} r={8} className={`sqd__dot sqd__dot--${cls}`} />
              )}
              {(() => {
                // beside the mark when there is little room above it (narrow layout), else above it
                const vx = x(r.value);
                const nearRight = vx + 70 > width;
                const beside = bars || compact;
                const tx = beside ? (nearRight ? vx - 12 : vx + 12) : vx;
                return (
                  <text x={tx} y={beside ? cy + 4 : cy - 14} textAnchor={beside ? (nearRight ? "end" : "start") : "middle"} className="sqd__val num">
                    {r.text ?? mha(r.value)}
                  </text>
                );
              })()}
            </g>
          );
        })}
        <text x={width - right} y={height - 4} textAnchor="end" className="chart__axis">{axis}</text>
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ the three views of one run


/** Without an exact energy there is no error to plot, so show how far below Hartree-Fock each method gets. */
function GainComparison({ r }: { r: SqdRow }) {
  const ccsdGain = r.below_hf_mha !== null && r.below_ccsd_mha !== null ? r.below_hf_mha - r.below_ccsd_mha : null;
  const bars = [
    { key: "hf", label: "Hartree-Fock", sub: "classical, cheapest", value: 0, tone: "hf" },
    { key: "cc", label: "CCSD", sub: "classical, standard method", value: ccsdGain ?? 0, tone: "cc" },
    { key: "sqd", label: "SQD", sub: "circuit samples + classical solve", value: r.below_hf_mha ?? 0, tone: "q" },
  ];
  const max = Math.max(1, ...bars.map((b) => Math.abs(b.value)));
  const sampling = r.sampling_time_s;
  const diag = r.diagonalisation_time_s;
  const total = sampling !== null && diag !== null ? sampling + diag : null;
  return (
    <Panel title="Classical methods against SQD" aside={`${molName(r.molecule)}, ${r.qubits} qubits, no exact energy at this size`}>
      <div className="sqd__gain">
        {bars.map((b) => (
          <div key={b.key} className="sqd__gainrow">
            <div>
              <div className="sqd__row2">{b.label}</div>
              <div className="small faint">{b.sub}</div>
            </div>
            <div className="sqd__gaintrack">
              <div className={`sqd__gainbar sqd__gainbar--${b.tone}`} style={{ width: `${Math.max(0, (100 * b.value) / max)}%` }} />
            </div>
            <div className="num sqd__gainval">{b.value === 0 && b.key === "hf" ? "baseline" : `${mha(b.value)} mHa`}</div>
          </div>
        ))}
      </div>
      <p className="small faint" style={{ marginTop: 4 }}>Bars: how far the energy is below Hartree-Fock. Longer is better.</p>

      <div className="tablewrap" style={{ marginTop: 14 }}>
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">What runs where</th>
              <th scope="col" className="right">Energy (Ha)</th>
              <th scope="col" className="right">Below Hartree-Fock (mHa)</th>
              <th scope="col" className="right">Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Hartree-Fock</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">{ha(r.e_hf)}</td>
              <td className="right num">0</td>
              <td className="right num">instant</td>
            </tr>
            <tr>
              <td>CCSD</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">{ha(r.e_ccsd)}</td>
              <td className="right num">{mha(ccsdGain)}</td>
              <td className="right num">{secs(r.ccsd_time_s)}</td>
            </tr>
            <tr className="is-best">
              <td>SQD</td>
              <td>
                <span className="sqd__where sqd__where--q">{r.hardware ? "real device" : "circuit sampling"}</span> <span className="sqd__where">classical solve</span>
              </td>
              <td className="right num">{ha(r.e_sqd)}</td>
              <td className="right num">{mha(r.below_hf_mha)}</td>
              <td className="right num">
                {secs(total)}
                {total !== null && <span className="small faint"> ({secs(sampling)} + {secs(diag)})</span>}
              </td>
            </tr>
            <tr>
              <td>Exact (FCI)</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">not computed</td>
              <td className="right num">n/a</td>
              <td className="right num">infeasible</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="small muted" style={{ marginTop: 12, maxWidth: "74ch" }}>
        This problem has {int(r.n_determinants)} determinants, too many to solve exactly, so there is no reference to measure an error
        against. SQD&apos;s energy comes from an exact solve inside a smaller subspace ({int(r.subspace_dim)} determinants), which makes it an
        upper bound on the true ground-state energy: a lower energy is a better answer. {r.below_ccsd_mha !== null && r.below_ccsd_mha > 0 ? "It is lower than CCSD here." : "It is not lower than CCSD here, so it does not beat the classical method at this size."}
      </p>
    </Panel>
  );
}

function EnergyRounds({ r }: { r: SqdRow }) {
  return (
    <Panel title="Energy after each round" aside="the lowest is kept">
      <div className="tablewrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Round</th>
              <th scope="col" className="right">Energy (Ha)</th>
              <th scope="col" className="right">Below Hartree-Fock (mHa)</th>
            </tr>
          </thead>
          <tbody>
            {r.energy_per_iteration.map((e, i) => (
              <tr key={i}>
                <td>Round {i + 1}</td>
                <td className="right num">{ha(e)}</td>
                <td className="right num">{r.e_hf !== null ? mha((r.e_hf - e) * 1000) : "n/a"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>
        Rounds do not always improve: each one draws new random batches, so the energy can move up or down a little.
      </p>
    </Panel>
  );
}

function HardwareInfo({ r }: { r: SqdRow }) {
  const hw = r.hardware;
  if (!hw) return null;
  const ideal = hw.ideal_same_circuit;
  return (
    <Panel title="This run used a real quantum computer" aside={hw.backend}>
      <div className="tablewrap">
        <table className="table">
          <tbody>
            <tr><td>Device</td><td className="right num">{hw.backend}</td></tr>
            <tr><td>IBM job</td><td className="right num small">{hw.job_id}</td></tr>
            <tr><td>Shots</td><td className="right num">{int(hw.shots ?? r.shots)}</td></tr>
            <tr><td>Circuit on the device</td><td className="right num">{int(hw.two_qubit_gates)} two-qubit gates, depth {int(hw.depth)}, {int(hw.qubits_used)} qubits</td></tr>
            <tr><td>QPU time used</td><td className="right num">{hw.qpu_seconds === null ? "not reported" : secs(hw.qpu_seconds)}{hw.estimated_qpu_s !== undefined && <span className="small faint"> (estimated {secs(hw.estimated_qpu_s)})</span>}</td></tr>
            {ideal && (
              <tr>
                <td>Same circuit on an ideal simulator</td>
                <td className="right num">
                  {ideal.error_mha !== null ? `${mha(ideal.error_mha)} mHa error` : `${mha(ideal.below_hf_mha)} mHa below Hartree-Fock`}, {int(ideal.distinct_samples)} distinct bitstrings
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 10, maxWidth: "74ch" }}>
        The bitstrings came from the device; recovery, post-selection and the subspace solve were done on this computer. The circuit
        uses device-friendly couplings, so it differs from the simulated runs above.
        {ideal
          ? ` It gave ${int(r.distinct_samples)} distinct bitstrings on the device against ${int(ideal.distinct_samples)} on an ideal simulator${
              (r.distinct_samples ?? 0) > ideal.distinct_samples * 1.2 ? ": device noise widens the samples, which helps SQD." : "."
            }`
          : ""}
      </p>
    </Panel>
  );
}

function Stats({ run }: { run: SqdRow }) {
  const exact = run.error_mha !== null;
  return (
    <div className="stats">
      <Stat label="Qubits" value={run.qubits ?? "n/a"} note={`${molName(run.molecule)} at ${run.bond_length} Å`} />
      {exact ? (
        <>
          <Stat label="SQD error" value={mha(run.error_mha)} unit="mHa" note={run.chem_acc ? "within chemical accuracy" : `chemical accuracy is ${CHEM_ACC_MHA}`} />
          <Stat label="Hartree-Fock error" value={mha(run.hf_error_mha)} unit="mHa" note="classical starting point" />
          <Stat label="CCSD error" value={mha(run.ccsd_error_mha)} unit="mHa" note="classical standard" />
        </>
      ) : (
        <>
          <Stat label="Below Hartree-Fock" value={mha(run.below_hf_mha)} unit="mHa" note="no exact energy at this size" />
          <Stat label="Below CCSD" value={mha(run.below_ccsd_mha)} unit="mHa" note={(run.below_ccsd_mha ?? 0) > 0 ? "lower than the classical method" : "not lower than the classical method"} />
        </>
      )}
      <Stat label="Subspace solved" value={run.subspace_fraction !== null ? `${(100 * run.subspace_fraction).toFixed(run.subspace_fraction < 0.001 ? 4 : 0)}%` : "n/a"} note={`${int(run.subspace_dim)} of ${int(run.n_determinants)}`} />
    </div>
  );
}

function runLabel(r: SqdRow): string {
  const where = r.hardware ? `on ${r.hardware.backend}, ${r.shots} shots` : `noise ${r.readout_noise}, ${r.iterations} rounds, seed ${r.seed}`;
  const score = r.error_mha !== null ? `${mha(r.error_mha)} mHa error` : `${mha(r.below_hf_mha)} mHa below HF`;
  return `${molName(r.molecule)} ${where}: ${score}`;
}

/** Measured: 3 to 6 s at 20 qubits, 2 to 5 s at 24, 4 to 19 s at 28, 25 to 60 s at 32. */
function simulateSeconds(f: SqdRunRequest, determinants: number): number {
  return 2 + f.shots / 10000 + f.iterations * f.num_batches * 0.15 + determinants * 1.3e-6 * f.n_reps;
}

const DEVICE_MODELS = ["FakeFez", "FakeSherbrooke", "FakeBrisbane", "FakeKyiv", "FakeTorino"];

function Comparison({ r }: { r: SqdRow }) {
  if (r.error_mha === null) return <GainComparison r={r} />;
  const rows: LogRow[] = [
    ...(r.hf_error_mha !== null ? [{ key: "hf", label: "Hartree-Fock", sub: "classical, cheapest", value: r.hf_error_mha, tone: "hf" as const }] : []),
    ...(r.ccsd_error_mha !== null ? [{ key: "cc", label: "CCSD", sub: "classical, standard method", value: r.ccsd_error_mha, tone: "cc" as const }] : []),
    { key: "sqd", label: "SQD", sub: "circuit samples + classical solve", value: r.error_mha, tone: "q" },
  ];
  const sampling = r.sampling_time_s;
  const diag = r.diagonalisation_time_s;
  const total = sampling !== null && diag !== null ? sampling + diag : null;
  return (
    <Panel title="Classical methods against SQD" aside={`${molName(r.molecule)}, ${r.qubits} qubits, error against the exact energy`}>
      <LogRows rows={rows} domain={[0.1, 1000]} band axis="error in mHa, log scale" aria="Error of Hartree-Fock, CCSD and SQD against the exact energy" />
      <div className="chart__key small" style={{ marginTop: 6 }}>
        <span><i className="sw sw--ok" /> within chemical accuracy ({CHEM_ACC_MHA} mHa)</span>
      </div>

      <div className="tablewrap" style={{ marginTop: 14 }}>
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">What runs where</th>
              <th scope="col" className="right">Energy (Ha)</th>
              <th scope="col" className="right">Error (mHa)</th>
              <th scope="col" className="right">Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Hartree-Fock</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">{ha(r.e_hf)}</td>
              <td className="right num">{mha(r.hf_error_mha)}</td>
              <td className="right num">instant</td>
            </tr>
            <tr>
              <td>CCSD</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">{ha(r.e_ccsd)}</td>
              <td className="right num">{mha(r.ccsd_error_mha)}</td>
              <td className="right num">{secs(r.ccsd_time_s)}</td>
            </tr>
            <tr className="is-best">
              <td>SQD</td>
              <td>
                <span className="sqd__where sqd__where--q">circuit sampling</span> <span className="sqd__where">classical solve</span>
              </td>
              <td className="right num">{ha(r.e_sqd)}</td>
              <td className="right num">{mha(r.error_mha)}</td>
              <td className="right num">
                {secs(total)}
                {total !== null && <span className="small faint"> ({secs(sampling)} + {secs(diag)})</span>}
              </td>
            </tr>
            <tr>
              <td>Exact (FCI)</td>
              <td><span className="sqd__where">classical</span></td>
              <td className="right num">{ha(r.e_exact)}</td>
              <td className="right num">0 (reference)</td>
              <td className="right num">{secs(r.fci_time_s)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="small muted" style={{ marginTop: 12, maxWidth: "74ch" }}>
        At 20 qubits every method here finishes in seconds: the classical methods are as fast and about as accurate. SQD is
        not shown to be better at this size. What it changes is the scaling: its classical step solves a small subspace
        ({int(r.subspace_dim)} of {int(r.n_determinants)} determinants here) instead of the whole space, so it can reach active
        spaces too large to solve exactly. {r.hardware ? `The circuit-sampling step ran on ${r.hardware.backend}.` : "The circuit-sampling step is simulated on this computer; on a quantum device it would be the device."}
      </p>
    </Panel>
  );
}

function Flow({ r }: { r: SqdRow }) {
  return (
    <div className="sqd__flow">
      <div className="sqd__step sqd__step--q">
        <span className="sqd__tag">1 · quantum-circuit step</span>
        <strong style={{ fontWeight: 400 }}>Sample the circuit</strong>
        <span className="small muted">{int(r.shots)} shots of a LUCJ circuit on {r.qubits} qubits ({r.hardware ? `real device ${r.hardware.backend}` : "simulated here"}).</span>
      </div>
      <div className="sqd__step">
        <span className="sqd__tag">2 · classical</span>
        <strong style={{ fontWeight: 400 }}>Recover configurations</strong>
        <span className="small muted">Repair bitstrings with the wrong electron count.</span>
      </div>
      <div className="sqd__step">
        <span className="sqd__tag">3 · classical</span>
        <strong style={{ fontWeight: 400 }}>Post-select and subsample</strong>
        <span className="small muted">{int(r.postselected)} usable bitstrings, {r.samples_per_batch} per batch.</span>
      </div>
      <div className="sqd__step">
        <span className="sqd__tag">4 · classical</span>
        <strong style={{ fontWeight: 400 }}>Diagonalise the subspace</strong>
        <span className="small muted">Exact solve in a {int(r.subspace_dim)}-determinant subspace, repeated for {r.iterations} rounds.</span>
      </div>
    </div>
  );
}

function Rounds({ r }: { r: SqdRow }) {
  if (r.error_per_iteration_mha.length === 0) return <EnergyRounds r={r} />;
  const rows: LogRow[] = r.error_per_iteration_mha.map((e, i) => ({
    key: `r${i}`,
    label: `Round ${i + 1}`,
    sub: i === 0 ? "recovery from the raw samples" : "recovery from the last round",
    value: e,
    tone: "q",
  }));
  return (
    <Panel title="Error after each round" aside="the best round is kept">
      <LogRows rows={rows} domain={[0.1, 1000]} band axis="error in mHa, log scale" aria="SQD error after each round" />
      <p className="small muted" style={{ marginTop: 8 }}>
        Rounds do not always improve: each one draws new random batches, so the error can move up or down a little.
      </p>
    </Panel>
  );
}

function Funnel({ r }: { r: SqdRow }) {
  const rows: LogRow[] = [
    { key: "shots", label: "Shots", sub: "bitstrings sampled", value: r.shots, tone: "cc", text: int(r.shots) },
    { key: "distinct", label: "Distinct bitstrings", value: r.distinct_samples ?? 1, tone: "cc", text: int(r.distinct_samples) },
    { key: "post", label: "Right electron count", sub: "after recovery", value: r.postselected ?? 1, tone: "cc", text: int(r.postselected) },
    { key: "sub", label: "Subspace solved", sub: "determinants", value: r.subspace_dim ?? 1, tone: "q", text: int(r.subspace_dim) },
    { key: "full", label: "Full problem", sub: "all determinants", value: r.n_determinants ?? 1, tone: "full", text: int(r.n_determinants) },
  ];
  return (
    <Panel title="How much of the problem is solved" aside={r.subspace_fraction !== null ? `${(100 * r.subspace_fraction).toFixed(0)}% of the full space` : undefined}>
      <LogRows rows={rows} domain={[1, 100000]} axis="count, log scale" bars aria="From shots to the subspace that is solved" />
    </Panel>
  );
}

// ------------------------------------------------------------------ across seeds

/** What the seed study says, computed from the data so it can never drift from the chart. */
function Takeaways({ groups }: { groups: SqdGroup[] }) {
  const best = [...groups].sort((a, b) => a.mean_error_mha - b.mean_error_mha)[0];
  const clean = groups.find((g) => g.readout_noise === 0);
  const base = groups.find(
    (g) => g.readout_noise === DEFAULTS.readout_noise && g.samples_per_batch === DEFAULTS.samples_per_batch && g.iterations === DEFAULTS.iterations && g.n_reps === DEFAULTS.n_reps,
  );
  const saturated = base
    ? groups.find(
        (g) =>
          g !== base &&
          g.samples_per_batch > base.samples_per_batch &&
          g.readout_noise === base.readout_noise &&
          g.iterations === base.iterations &&
          g.n_reps === base.n_reps &&
          Math.abs(g.mean_error_mha - base.mean_error_mha) < 1e-9,
      )
    : undefined;
  const within = (g: SqdGroup) => `${Math.round(g.share_chem_acc * g.n_seeds)} of ${g.n_seeds} seeds within ${CHEM_ACC_MHA} mHa`;
  return (
    <ul className="small muted" style={{ margin: "0 0 12px", paddingLeft: 18, display: "grid", gap: 4, maxWidth: "78ch" }}>
      {best && (
        <li>
          Best setting: <strong>{groupLabel(best)}</strong>, {mha(best.mean_error_mha)} ± {mha(best.std_error_mha)} mHa on average ({within(best)}). Classical CCSD is {mha(best.ccsd_error_mha)} mHa.
        </li>
      )}
      {base && base !== best && (
        <li>
          Default setting: {mha(base.mean_error_mha)} ± {mha(base.std_error_mha)} mHa ({within(base)}).
        </li>
      )}
      {clean && (
        <li>
          Without sampling noise: {mha(clean.mean_error_mha)} mHa, because the circuit alone gives only a few dozen distinct bitstrings.
        </li>
      )}
      {saturated && (
        <li>
          {saturated.samples_per_batch} samples per batch gives the same result as {base?.samples_per_batch}: the batch cannot hold more bitstrings than have the right
          electron count, so it never fills.
        </li>
      )}
    </ul>
  );
}

function Strips({ groups }: { groups: SqdGroup[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const compact = width < 600;
  const domain: [number, number] = [0.2, 1000];
  const left = compact ? 14 : 290;
  const right = compact ? 14 : 190;
  const rowH = compact ? 78 : 44;
  const top = 26;
  const x = log(domain, [left, Math.max(left + 120, width - right)]);
  const height = top + groups.length * rowH + 40;
  const bottom = top - 8 + groups.length * rowH;
  const first = groups[0];
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={height} role="img" aria-label="SQD error for each seed, by setting">
        <rect x={x(domain[0])} y={top - 8} width={x(CHEM_ACC_MHA) - x(domain[0])} height={groups.length * rowH} className="chart__band" />
        {first?.hf_error_mha ? <line x1={x(first.hf_error_mha)} x2={x(first.hf_error_mha)} y1={top - 8} y2={bottom} className="chart__hf" /> : null}
        {first?.ccsd_error_mha ? <line x1={x(first.ccsd_error_mha)} x2={x(first.ccsd_error_mha)} y1={top - 8} y2={bottom} className="chart__exact" strokeDasharray="2 4" /> : null}
        {[1, 10, 100, 1000].map((t) => (
          <text key={t} x={x(t)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{t}</text>
        ))}
        {groups.map((g, k) => {
          const cy = top + k * rowH + (compact ? rowH - 22 : rowH / 2 - 6);
          const lx = compact ? left : left - 14;
          return (
            <g key={groupLabel(g) + g.bond_length}>
              <text x={lx} y={compact ? cy - 40 : cy + 4} textAnchor={compact ? "start" : "end"} className="sqd__row">{groupLabel(g)}</text>
              {compact && (
                <text x={lx} y={cy - 24} className="sqd__label num">
                  mean {mha(g.mean_error_mha)} ± {mha(g.std_error_mha)} · {Math.round(g.share_chem_acc * g.n_seeds)} of {g.n_seeds} within {CHEM_ACC_MHA}
                </text>
              )}
              <line x1={x(domain[0])} x2={x(domain[1])} y1={cy} y2={cy} className="sqd__track" />
              {g.errors.map((e, i) => (
                <circle key={i} cx={x(e)} cy={cy} r={4.5} className={`sqd__seed ${e <= CHEM_ACC_MHA ? "sqd__seed--ok" : ""}`} />
              ))}
              <line x1={x(g.mean_error_mha)} x2={x(g.mean_error_mha)} y1={cy - 11} y2={cy + 11} className="sqd__mean" />
              {!compact && (
                <text x={width - right + 14} y={cy + 4} className="sqd__label num">
                  {mha(g.mean_error_mha)} ± {mha(g.std_error_mha)} · {Math.round(g.share_chem_acc * g.n_seeds)}/{g.n_seeds} ≤ {CHEM_ACC_MHA}
                </text>
              )}
            </g>
          );
        })}
        <text x={x(domain[1])} y={height - 4} textAnchor="end" className="chart__axis">error in mHa, log scale; each dot is one seed, the bar is the mean</text>
      </svg>
      <div className="chart__key small">
        <span><i className="sw sw--ok" /> within chemical accuracy</span>
        <span><i className="sw sw--hf" /> Hartree-Fock ({mha(first?.hf_error_mha)})</span>
        <span><i className="sw sw--exact" /> CCSD ({mha(first?.ccsd_error_mha)})</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ starting a run

const DEFAULTS: SqdRunRequest = {
  molecule: "n2",
  bond_length: null,
  n_reps: 1,
  shots: 20000,
  readout_noise: 0.02,
  noise_model: null,
  exact: false,
  iterations: 3,
  samples_per_batch: 300,
  num_batches: 3,
  seed: 42,
};

function RunForm({ onStarted }: { onStarted: (id: string) => void }) {
  const opts = useQuery<SqdOptions>("/api/sqd/options");
  const { job } = useJobs();
  const [f, setF] = useState<SqdRunRequest>(DEFAULTS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = <K extends keyof SqdRunRequest>(k: K, v: SqdRunRequest[K]) => setF((p) => ({ ...p, [k]: v }));
  const cur = job(jobId);
  const running = cur && (cur.status === "queued" || cur.status === "running");

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      const j = await api.post<Job>("/api/sqd/run", f);
      setJobId(j.id);
      onStarted(j.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const num = (k: keyof SqdRunRequest, step?: string) => (
    <input className="input num" type="number" step={step} value={(f[k] as number | null) ?? ""} onChange={(e) => set(k, Number(e.target.value) as never)} />
  );
  const mol = opts.data?.molecules.find((m) => m.id === f.molecule);

  return (
    <Panel title="Run SQD" aside="saved with your runs">
      <div className="form">
        {opts.data && (
          <Select
            label="Molecule"
            value={f.molecule}
            onChange={(v) => set("molecule", v)}
            options={opts.data.molecules.map((m) => ({
              id: m.id,
              label: `${molName(m.name)}, ${m.qubits} qubits${m.simulable ? "" : " (real hardware only)"}`,
            }))}
            hint={mol?.simulable === false ? "Too large to simulate here: use the real-hardware section below." : mol?.description}
          />
        )}
        <div className="form__row">
          <Select
            label="Sampling noise"
            value={f.noise_model ?? ""}
            onChange={(v) => set("noise_model", v || null)}
            options={[{ id: "", label: "Uniform bit flips (simple)" }, ...DEVICE_MODELS.map((d) => ({ id: d, label: `${d} calibration (readout and gate errors)` }))]}
            hint={f.noise_model ? "Per-qubit readout errors and the chance a shot survives every gate, from a fake IBM device. An approximation, not a real device." : "Every sampled bit flips with the same chance."}
          />
          <Field label="Bit-flip chance" hint={f.noise_model ? "not used with a device model" : "chance each sampled bit is flipped"}>
            <input className="input num" type="number" step="0.01" disabled={!!f.noise_model} value={f.readout_noise} onChange={(e) => set("readout_noise", Number(e.target.value))} />
          </Field>
        </div>
        <div className="form__row">
          <Field label="Shots" hint="bitstrings sampled">{num("shots", "1000")}</Field>
          <Field label="Bond length (Å)" hint={mol ? `empty = ${mol.default_bond_length}` : undefined}>
            <input
              className="input num"
              type="number"
              step="0.01"
              placeholder={mol ? String(mol.default_bond_length) : ""}
              value={f.bond_length ?? ""}
              onChange={(e) => set("bond_length", e.target.value === "" ? null : Number(e.target.value))}
            />
          </Field>
        </div>
        <div className="form__row">
          <Field label="Rounds" hint="recovery and diagonalisation rounds">{num("iterations")}</Field>
          <Field label="Samples per batch" hint="bitstrings used for each subspace">{num("samples_per_batch", "50")}</Field>
        </div>
        <div className="form__row">
          <Field label="Circuit layers" hint="LUCJ layers">{num("n_reps")}</Field>
          <Field label="Random seed">{num("seed")}</Field>
        </div>
        {mol && mol.simulable && !mol.has_exact && (
          <label className="small" style={{ display: "flex", gap: 8, alignItems: "flex-start", maxWidth: "70ch" }}>
            <input type="checkbox" checked={f.exact} onChange={(e) => set("exact", e.target.checked)} style={{ marginTop: 3 }} />
            <span>
              Also solve the exact energy, so the result shows an error and not only the gain over Hartree-Fock. This takes about{" "}
              {duration(mol.exact_estimate_s)} the first time and is saved, so it never has to run again.
            </span>
          </label>
        )}
        {mol?.exact_cached && mol.simulable && <p className="small muted">The exact energy for this molecule is already saved, so the result will show an error.</p>}
        <div className="form__submit">
          <button type="button" className="btn" disabled={busy || !!running || mol?.simulable === false} onClick={submit}>
            {running ? "Running…" : "Run SQD"}
          </button>
          <span className="small muted">
            Estimated time: about {duration(simulateSeconds(f, mol?.determinants ?? 14400) + (f.exact && mol && !mol.has_exact ? mol.exact_estimate_s : 0))}. Sampling is simulated and never uses a quantum device.
          </span>
        </div>
        {err && <Notice tone="bad">{err}</Notice>}
        {cur?.status === "failed" && <Notice tone="bad">{cur.error}</Notice>}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ page

export function SqdPage() {
  const { source, setSource } = useSource();
  const { finishedCount, job } = useJobs();
  const q = useQuery<SqdResponse>(`/api/sqd${qs({ source, n: finishedCount })}`);
  const [sel, setSel] = useState<string | null>(null);
  const [watch, setWatch] = useState<string | null>(null);
  const [molSel, setMolSel] = useState<string | null>(null);

  // when a run started here finishes, switch to your runs and show it
  const watched = job(watch);
  useEffect(() => {
    if (watched?.status === "done" && watched.result_hashes[0]) {
      setSource("local");
      setSel(watched.result_hashes[0]);
      setWatch(null);
    }
  }, [watched?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const allRows = useMemo(() => q.data?.runs ?? [], [q.data]);
  const allGroups = useMemo(() => q.data?.groups ?? [], [q.data]);
  const molecules = useMemo(() => [...new Set(allRows.filter((r) => r.status === "ok").map((r) => r.molecule))].sort(), [allRows]);
  // the molecule shown: the one a selected or just-finished run belongs to, else the user's pick, else N2
  const molecule = allRows.find((r) => r.hash === sel)?.molecule ?? (molSel && molecules.includes(molSel) ? molSel : molecules.includes("N2") ? "N2" : molecules[0]);
  const rows = useMemo(() => allRows.filter((r) => r.molecule === molecule), [allRows, molecule]);
  const groups = useMemo(() => allGroups.filter((g) => g.molecule === molecule), [allGroups, molecule]);
  const run = useMemo(
    () => rows.find((r) => r.hash === sel && r.status === "ok") ?? typicalRun(rows, groups),
    [rows, groups, sel],
  );

  if (q.loading && !q.data) return <div className="page"><Loading /></div>;
  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;

  const ok = rows.filter((r) => r.status === "ok");
  return (
    <div className="page">
      <PageHead
        title="SQD: larger molecules"
        sub="Sample-based quantum diagonalization reaches active spaces far beyond the 2 to 6 qubits of VQE: it samples bitstrings from a circuit, repairs them, and solves the molecule exactly inside the small subspace they span."
      />

      {molecules.length > 1 && (
        <div className="mv__chips" role="tablist" aria-label="Molecule shown" style={{ marginBottom: 14 }}>
          {molecules.map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={m === molecule}
              className={`mv__chip ${m === molecule ? "is-on" : ""}`}
              onClick={() => {
                setSel(null);
                setMolSel(m);
              }}
            >
              {molName(m)}
            </button>
          ))}
        </div>
      )}

      {run?.hardware ? (
        <Notice tone="warn">
          REAL DEVICE result from {run.hardware.backend}. The bitstrings were sampled on a quantum computer; everything after that
          ran on this computer. A single run is not a benchmark, and this is not a claim of quantum advantage.
        </Notice>
      ) : (
        <Notice tone="warn">
          The circuit sampling here is <strong>simulated</strong>.{" "}
          {run?.noise_model
            ? `The noise comes from the calibration of a fake IBM device (${run.noise_model}): readout errors per qubit and the chance a shot survives every gate. It approximates a device; it is not one.`
            : "Random bit flips stand in for measurement noise. They are not a real device and not a model of one."}{" "}
          These runs show the method working at {run?.qubits ?? 20} qubits {run?.error_mha !== null && run?.error_mha !== undefined ? "against an exact reference" : "without an exact reference"}, not a
          quantum advantage.
        </Notice>
      )}

      <MoleculeViewer focus={molecule} />

      {!run ? (
        <Panel>
          <Empty title="No SQD runs in this source" art="blocks">
            {DEMO
              ? "The offline demo has no SQD results."
              : "Run one below, or switch to the published results. From the command line: miraichem sqd --molecule configs/sqd/n2.yaml --readout-noise 0.02"}
          </Empty>
        </Panel>
      ) : (
        <>
          <Panel
            title="Selected run"
            aside={
              <span className="sqd__pick">
                <label className="small muted" htmlFor="sqd-run">Run</label>
                <select id="sqd-run" className="select" value={run.hash} onChange={(e) => setSel(e.target.value)}>
                  {ok.slice(0, 300).map((r) => (
                    <option key={r.hash} value={r.hash}>
                      {runLabel(r)}
                    </option>
                  ))}
                </select>
              </span>
            }
          >
            <Stats run={run} />
            <div style={{ height: 14 }} />
            <Flow r={run} />
            <p className="small faint">
              {run.hardware ? "Shown: a run on a real device." : "Shown: the median seed of the default setting. The seed study below shows every seed and every setting."}
            </p>
          </Panel>

          <HardwareInfo r={run} />

          <Comparison r={run} />

          <div className="grid grid--2">
            <Rounds r={run} />
            <Funnel r={run} />
          </div>
        </>
      )}

      {groups.length > 0 && (
        <Panel title="Across seeds" aside={`${ok.length} runs, ${groups.length} settings`}>
          <p className="small muted" style={{ marginBottom: 10, maxWidth: "76ch" }}>
            One run can be lucky, so each setting is repeated over several seeds. Noise-free samples give too few distinct
            bitstrings to span a useful subspace; a little noise widens them, which is what the lower rows show.
          </p>
          <Takeaways groups={groups} />
          <Strips groups={groups} />
        </Panel>
      )}

      <SqdScaling rows={allRows} groups={allGroups} />

      {!DEMO && <RunForm onStarted={setWatch} />}
      {!DEMO && (
        <SqdHardware
          onFinished={(hash) => {
            setSource("local");
            setSel(hash);
          }}
        />
      )}
    </div>
  );
}
