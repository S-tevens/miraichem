import { useState } from "react";
import { qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import { log, logTicks } from "../lib/scale";
import { MITIGATION, mha, moleculeName } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { CopyButton, Empty, Loading, Notice, PageHead, Panel, Seg } from "../ui/ui";
import { CHEM_ACC_MHA, type HardwareResult } from "../types";
import "../components/charts.css";
import "./Hardware.css";

const ORDER = ["none", "readout", "zne", "readout+zne"];

function runName(h: HardwareResult): string {
  const l = (h.source_label || h.source)
    .replace(/^uccsd\(reps=1\)/, "UCCSD")
    .replace(/^hea\(reps=(\d)\)/, "Hardware-efficient, $1 layers")
    .replace(/\/spsa/, " · SPSA")
    .replace(/\/cobyla/, " · COBYLA")
    .replace(/jordan_wigner/, "Jordan-Wigner")
    .replace(/parity/, "Parity");
  return `${l} on ${h.device}`;
}

/** The hardware run with all four mitigation modes and the smallest ideal error. */
function mainRun(runs: HardwareResult[]): number {
  const k = [...runs.keys()]
    .filter((i) => runs[i].evaluations.length >= 4)
    .sort((a, b) => runs[a].ideal_error_mha - runs[b].ideal_error_mha)[0];
  return k ?? 0;
}

const DRY_RUN = `miraichem hardware --molecule configs/molecules/h2.yaml --config-hash <hash> --shots 4096`;

export function HardwarePage() {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const q = useQuery<{ results: HardwareResult[] }>(`/api/hardware${qs({ source, n: finishedCount })}`);
  const runs = q.data?.results ?? [];
  const [i, setI] = useState<number | null>(null);
  const [ref, width] = useWidth<HTMLDivElement>();

  if (q.loading && !q.data) return <div className="page"><Loading /></div>;
  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;

  const gate = (
    <Panel title="How real-hardware runs are started">
      <div className="grid grid--2">
        <div>
          <p className="small muted" style={{ maxWidth: "58ch" }}>
            Real quantum time is scarce, so these VQE hardware runs are started only from the command line, which keeps three
            safeguards: an explicit <code>--allow-hardware</code> flag, a cost estimate shown first, and a yes/no
            confirmation. Job IDs are saved the moment a job is submitted. The parameters found in simulation are
            evaluated once per mitigation mode instead of optimising on the device. SQD runs on real hardware are started from
            the SQD page, after a plan and an &ldquo;are you sure&rdquo; step.
          </p>
        </div>
        <div>
          <p className="small muted" style={{ marginBottom: 6 }}>Dry run (prints the estimate and submits nothing):</p>
          <code className="code">{DRY_RUN}</code>
          <div style={{ marginTop: 8 }}>
            <CopyButton text={DRY_RUN} />
          </div>
        </div>
      </div>
    </Panel>
  );

  if (!runs.length) {
    return (
      <div className="page">
        <PageHead title="Real hardware" />
        <Panel>
          <Empty title="No hardware results in this source" art="blocks">
            Switch to the published results to see the saved IBM job records, or run a hardware evaluation from the command line.
          </Empty>
        </Panel>
        {gate}
      </div>
    );
  }

  const h = runs[Math.min(i ?? mainRun(runs), runs.length - 1)];
  const evals = [...h.evaluations].sort((a, b) => ORDER.indexOf(a.mitigation) - ORDER.indexOf(b.mitigation));

  const compact = width < 560;
  const domain: [number, number] = [0.2, 1000];
  const left = compact ? 14 : 132;
  const right = 24;
  const rowH = compact ? 66 : 54;
  const top = 34;
  const height = top + evals.length * rowH + 44;
  const x = log(domain, [left, Math.max(left + 160, width - right)]);
  const ticks = logTicks(domain).filter((t) => [1, 10, 100, 1000].includes(t) || t === 0.2);
  const plotBottom = top - 8 + evals.length * rowH;

  return (
    <div className="page">
      <PageHead
        title="Real hardware"
        sub="The best parameters found in simulation, evaluated once per mitigation mode on a real IBM device. These are single runs, so the bars show each run's statistical error, not a seed-to-seed spread."
        tools={
          runs.length > 1 ? (
            <Seg
              label="Hardware run"
              value={String(Math.min(i ?? mainRun(runs), runs.length - 1))}
              onChange={(v) => setI(Number(v))}
              options={runs.map((r, k) => ({ id: String(k), label: `${moleculeName(r.molecule)} · ${runName(r)}` }))}
            />
          ) : undefined
        }
      />

      <Notice tone="warn">
        REAL DEVICE results from {h.device}, {h.shots.toLocaleString("en")} shots per circuit, {h.timestamp.slice(0, 10)}.
        Simulator numbers elsewhere in this app are not hardware results.
      </Notice>

      <div className="grid grid--side hw">
        <Panel title="Error by mitigation mode" aside={`${moleculeName(h.molecule)} at ${h.bond_length} Å`}>
          <div ref={ref} className="chart">
            <svg width={width} height={height} role="img" aria-label={`Hardware error by mitigation mode on ${h.device}`}>
              <rect x={x(domain[0])} y={top - 8} width={x(CHEM_ACC_MHA) - x(domain[0])} height={evals.length * rowH} className="chart__band" />
              <line x1={x(h.hf_error_mha)} x2={x(h.hf_error_mha)} y1={top - 8} y2={plotBottom} className="chart__hf" />
              <line x1={x(h.simulator_error_mha)} x2={x(h.simulator_error_mha)} y1={top - 8} y2={plotBottom} className="hw__sim" />
              {evals.map((e, k) => {
                const cy = top + k * rowH + (compact ? rowH - 22 : rowH / 2 - 8);
                const lo = Math.max(domain[0], e.error_mha - e.std_mha);
                return (
                  <g key={e.job_id}>
                    <text x={compact ? left : left - 14} y={compact ? cy - 26 : cy + 5} textAnchor={compact ? "start" : "end"} className="hw__label">
                      {MITIGATION[e.mitigation] ?? e.mitigation}
                    </text>
                    <line x1={x(domain[0])} x2={x(domain[1])} y1={cy} y2={cy} className="hw__track" />
                    <line x1={x(lo)} x2={x(e.error_mha + e.std_mha)} y1={cy} y2={cy} className="hw__err" />
                    <circle cx={x(e.error_mha)} cy={cy} r={7.5} className="hw__dot" />
                    <text x={x(e.error_mha)} y={cy - 14} textAnchor="middle" className="hw__val num">{mha(e.error_mha)}</text>
                  </g>
                );
              })}
              {ticks.map((t) => (
                <text key={t} x={x(t)} y={plotBottom + 20} textAnchor="middle" className="chart__axis num">{t}</text>
              ))}
              <text x={width - right} y={height - 4} textAnchor="end" className="chart__axis">error in mHa, log scale</text>
            </svg>
            <div className="chart__key small">
              <span><i className="sw sw--ok" /> within chemical accuracy</span>
              <span><i className="sw sw--hf" /> Hartree-Fock ({mha(h.hf_error_mha)})</span>
              <span><i className="sw sw--sim" /> simulator prediction ({mha(h.simulator_error_mha)}, {MITIGATION[h.simulator_mitigation] ?? h.simulator_mitigation})</span>
            </div>
            <p className="small faint" style={{ marginTop: 8 }}>
              The circuit itself, with no noise, is off by {mha(h.ideal_error_mha)} mHa at these parameters.
            </p>
          </div>
        </Panel>

        <Panel title="Saved job records">
          <div className="tablewrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Mitigation</th>
                  <th scope="col" className="right">Error (mHa)</th>
                  <th scope="col">Job ID</th>
                </tr>
              </thead>
              <tbody>
                {evals.map((e) => (
                  <tr key={e.job_id}>
                    <td>{MITIGATION[e.mitigation] ?? e.mitigation}</td>
                    <td className="right num">{mha(e.error_mha)} ± {mha(e.std_mha)}</td>
                    <td className="num small faint">{e.job_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 12 }}>
            Without mitigation this run is off by {mha(evals.find((e) => e.mitigation === "none")?.error_mha)} mHa. Simulated
            devices tend to underestimate unmitigated error, so treat simulator numbers as optimistic until checked on hardware.
          </p>
        </Panel>
      </div>
      {gate}
    </div>
  );
}
