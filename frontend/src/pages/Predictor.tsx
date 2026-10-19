import { useState } from "react";
import { ApiError, api, qs, useQuery } from "../api";
import { useJobs } from "../app/state";
import { DEMO } from "../lib/demo";
import { duration, pct } from "../lib/format";
import { log, logTicks } from "../lib/scale";
import { useWidth } from "../lib/hooks";
import { Loading, Notice, PageHead, Panel, Progress } from "../ui/ui";
import type { Job, PredictorReport, PredictorStatus } from "../types";
import "../components/charts.css";
import "./Predictor.css";

const LABELS: Record<string, string> = {
  log_two_qubit_gates: "two-qubit gate count",
  log_depth: "circuit depth",
  qubits: "qubits",
  parameters: "circuit parameters",
  log2_shots: "shots per circuit",
  log_budget: "evaluation budget",
  readout: "readout correction",
  zne: "zero-noise extrapolation",
  opt_cobyla: "optimizer: COBYLA",
  opt_spsa: "optimizer: SPSA",
  opt_spsa_clip: "optimizer: clipped SPSA",
  ansatz_uccsd: "circuit: UCCSD",
  ansatz_hea: "circuit: hardware-efficient",
  ansatz_hea_hf: "circuit: hardware-efficient, HF start",
  layers: "circuit layers",
  parity: "Parity mapping",
  log_hf_error: "molecule's Hartree-Fock error",
  log_device_gate_error: "device two-qubit gate error",
  log_device_readout_error: "device readout error",
};

function Scatter({ report }: { report: PredictorReport }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const pts = report.points;
  const vals = pts.flatMap((p) => [10 ** p.predicted, 10 ** p.actual]);
  const dom: [number, number] = [10 ** Math.floor(Math.log10(Math.min(...vals))), 10 ** Math.ceil(Math.log10(Math.max(...vals)))];
  const size = Math.min(width, 520);
  const pad = 46;
  const x = log(dom, [pad, size - 14]);
  const y = log(dom, [size - pad, 14]);
  const ticks = logTicks(dom).filter((t) => [0.01, 0.1, 1, 10, 100, 1000, 10000].includes(t));
  const mols = report.molecules;
  return (
    <div ref={ref} className="chart">
      <svg width={size} height={size} role="img" aria-label="Predicted against measured error on unseen configurations">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad} x2={size - 14} y1={y(t)} y2={y(t)} className="chart__grid" />
            <line x1={x(t)} x2={x(t)} y1={14} y2={size - pad} className="chart__grid" />
            <text x={pad - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">{t}</text>
            <text x={x(t)} y={size - pad + 18} textAnchor="middle" className="chart__axis num">{t}</text>
          </g>
        ))}
        <line x1={x(dom[0])} y1={y(dom[0])} x2={x(dom[1])} y2={y(dom[1])} className="chart__exact" strokeDasharray="4 4" />
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={x(10 ** p.predicted)}
            cy={y(10 ** p.actual)}
            r={3.2}
            className={`scatter__dot ${mols.indexOf(p.molecule) === 1 ? "scatter__dot--b" : ""}`}
          />
        ))}
        <text x={(pad + size) / 2} y={size - 4} textAnchor="middle" className="chart__axis">predicted error (mHa)</text>
        <text x={12} y={size / 2} transform={`rotate(-90 12 ${size / 2})`} textAnchor="middle" className="chart__axis">measured error (mHa)</text>
      </svg>
      <div className="chart__key small">
        {mols.map((m, i) => (
          <span key={m}><i className={`dotkey ${i === 1 ? "dotkey--amber" : "dotkey--front"}`} /> {m === "H2" ? "H₂" : m}</span>
        ))}
        <span><i className="sw sw--exact" /> perfect prediction</span>
      </div>
    </div>
  );
}

export function PredictorPage() {
  const { jobs, finishedCount } = useJobs();
  const q = useQuery<PredictorStatus>(`/api/predictor${qs({ n: finishedCount })}`);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const training = jobs.find((j: Job) => j.kind === "train" && (j.status === "queued" || j.status === "running"));

  const train = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api.post<Job>("/api/predictor/train");
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;
  if (!q.data) return <div className="page"><Loading /></div>;
  const s = q.data;
  const rep = s.report;

  const trainBox = DEMO ? null : (
    <Panel title="Training">
      <div className="train">
        <p className="small muted" style={{ maxWidth: "60ch" }}>
          {rep
            ? `Trained ${rep.trained} on ${rep.n_runs.toLocaleString("en")} noisy runs from ${rep.n_configurations} configurations (${s.origin === "published" ? "the published model" : "your runs"}). `
            : "No predictor has been trained yet. "}
          {s.noisy_runs_available.toLocaleString("en")} noisy runs are available in your results.
          {s.newer_data ? " There are newer runs than the model has seen." : ""}
        </p>
        <button type="button" className="btn" disabled={busy || !!training || s.noisy_runs_available < 60} onClick={train}>
          {training ? "Training…" : rep ? "Retrain on my runs" : "Train on my runs"}
        </button>
        {s.noisy_runs_available < 60 && <p className="small faint">At least 60 noisy runs are needed. Run a sweep first.</p>}
        {training && (
          <div style={{ width: "100%" }}>
            <Progress value={0} total={1} />
            <p className="small muted" style={{ marginTop: 6 }}>{training.message} {duration(training.elapsed_s)}</p>
          </div>
        )}
        {err && <Notice tone="bad">{err}</Notice>}
      </div>
    </Panel>
  );

  if (!rep) {
    return (
      <div className="page">
        <PageHead title="Error predictor" sub="Forecast a noisy run's error before you spend any compute on it." />
        {trainBox}
      </div>
    );
  }

  const cv = rep.cross_validation;
  const bars = [
    { name: "Predictor", m: rep.cross_validation, strong: true },
    { name: "Per-molecule average", m: rep.baseline_per_molecule, strong: false },
    { name: "Overall average", m: rep.baseline_mean, strong: false },
  ];
  const maxFactor = Math.max(...bars.map((b) => b.m.typical_factor));
  const imp = Object.entries(rep.importance).slice(0, 10);
  const maxImp = Math.max(...imp.map(([, v]) => v), 1e-9);

  return (
    <div className="page">
      <PageHead
        title="Error predictor"
        sub="Before you run a configuration, the predictor forecasts its error from the circuit size on the device, the optimizer, the mitigation, the shots and the device's noise. It is checked only on configurations it has never seen."
      />

      <div className="stats">
        <div className="stat">
          <span className="stat__label">Typical prediction error</span>
          <span className="stat__value num">{cv.typical_factor.toFixed(2)}×</span>
          <span className="stat__note">the answer is usually within this factor</span>
        </div>
        <div className="stat">
          <span className="stat__label">Without the model</span>
          <span className="stat__value num">{rep.baseline_per_molecule.typical_factor.toFixed(1)}×</span>
          <span className="stat__note">guessing the molecule's average</span>
        </div>
        <div className="stat">
          <span className="stat__label">Explained variation (R²)</span>
          <span className="stat__value num">{cv.r2.toFixed(2)}</span>
          <span className="stat__note">on unseen configurations</span>
        </div>
        <div className="stat">
          <span className="stat__label">80% interval is honest</span>
          <span className="stat__value num">{pct(rep.interval.coverage)}</span>
          <span className="stat__note">of true errors landed inside it</span>
        </div>
      </div>

      <div className="grid grid--2">
        <Panel title="Predicted against measured" aside="unseen configurations">
          <Scatter report={rep} />
          <p className="small faint" style={{ marginTop: 8 }}>
            Each dot is a run whose configuration was held out while the model was trained, so none of
            its seeds were seen. Points on the dashed line are perfect predictions.
          </p>
        </Panel>

        <div className="stack">
          <Panel title="Against simply guessing">
            <ul className="cmp">
              {bars.map((b) => (
                <li key={b.name}>
                  <span className="cmp__name">{b.name}</span>
                  <span className="cmp__bar">
                    <span className={`cmp__fill ${b.strong ? "cmp__fill--strong" : ""}`} style={{ width: `${(100 * b.m.typical_factor) / maxFactor}%` }} />
                  </span>
                  <span className="num cmp__val">{b.m.typical_factor.toFixed(2)}×</span>
                </li>
              ))}
            </ul>
            <p className="small faint">Typical factor between predicted and measured error; lower is better.</p>
          </Panel>

          <Panel title="By molecule">
            <div className="tablewrap">
              <table className="table">
                <thead>
                  <tr><th scope="col">Molecule</th><th scope="col" className="right">Runs</th><th scope="col" className="right">Typical factor</th><th scope="col" className="right">R²</th></tr>
                </thead>
                <tbody>
                  {Object.entries(rep.per_molecule).map(([m, v]) => (
                    <tr key={m}>
                      <td>{m === "H2" ? "H₂" : m}</td>
                      <td className="right num">{v.n}</td>
                      <td className="right num">{v.typical_factor.toFixed(2)}×</td>
                      <td className="right num">{v.r2.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--2">
        <Panel title="A harder test: a device it never saw" aside="the device is left out of training">
          <div className="tablewrap">
            <table className="table">
              <thead>
                <tr><th scope="col">Held-out device</th><th scope="col" className="right">Runs</th><th scope="col" className="right">Typical factor</th><th scope="col">Fair test?</th></tr>
              </thead>
              <tbody>
                {rep.device_holdout.map((h) => (
                  <tr key={h.device}>
                    <td>{h.device.replace("Fake", "")}</td>
                    <td className="right num">{h.n}</td>
                    <td className="right num">{h.typical_factor.toFixed(2)}×</td>
                    <td className="small muted">{h.fair ? "yes" : "no: most data came from this device"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small faint" style={{ marginTop: 10 }}>
            The model transfers to devices it never saw, though less precisely. A device that supplies most of the
            training data cannot be held out fairly, so that row is shown but not counted.
          </p>
        </Panel>

        <Panel title="What it relies on" aside="error added when an input is scrambled">
          <ul className="cmp">
            {imp.map(([k, v]) => (
              <li key={k}>
                <span className="cmp__name">{LABELS[k] ?? k}</span>
                <span className="cmp__bar"><span className="cmp__fill" style={{ width: `${(100 * Math.max(v, 0)) / maxImp}%` }} /></span>
                <span className="num cmp__val">+{Math.max(v, 0).toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel title="What it can and cannot do">
        <ul className="notes">
          <li>It was trained on {rep.molecules.map((m) => (m === "H2" ? "H₂" : m)).join(" and ")} on {rep.devices.length} simulated IBM device{rep.devices.length === 1 ? "" : "s"}. Anything outside that gets a warning and should be read as a guess.</li>
          <li>It predicts the error of one run with a random seed. Two runs of the same configuration differ, which is why there is an interval.</li>
          <li>It forecasts the noisy simulator, not real hardware, and not ideal runs.</li>
          <li>A typical error of {cv.typical_factor.toFixed(1)}× means it ranks options well and rules out clearly bad ones, but it will not replace running the configuration you care about.</li>
        </ul>
      </Panel>

      {trainBox}
    </div>
  );
}
