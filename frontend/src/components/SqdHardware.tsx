import { useEffect, useState } from "react";
import { ApiError, api, useQuery } from "../api";
import { useJobs } from "../app/state";
import { duration } from "../lib/format";
import { CopyButton, Field, Notice, Panel, Select } from "../ui/ui";
import type { Job, SqdHardwarePlan, SqdHardwareStatus, SqdOptions } from "../types";
import "../pages/NewRun.css";

const SUB = "₀₁₂₃₄₅₆₇₈₉";
/** "N2" -> "N₂"; "N2_631G" -> "N₂ (631G)" (the part after the underscore is the basis set). */
const molName = (n: string) => {
  const [head, ...rest] = n.split("_");
  const f = head.replace(/\d/g, (d) => SUB[Number(d)]);
  return rest.length ? `${f} (${rest.join(" ")})` : f;
};

interface Form {
  molecule: string;
  backend: string;
  shots: number;
  iterations: number;
  samples_per_batch: number;
  n_reps: number;
  seed: number;
}

const DEFAULTS: Form = { molecule: "n2", backend: "", shots: 10000, iterations: 3, samples_per_batch: 300, n_reps: 1, seed: 42 };

const secs = (s: number) => (s < 10 ? `${s.toFixed(1)} s` : duration(s));

/** Real-device SQD: plan first (uses no QPU time), then a typed confirmation, then the job. */
export function SqdHardware({ onFinished }: { onFinished: (hash: string) => void }) {
  const status = useQuery<SqdHardwareStatus>("/api/sqd/hardware");
  const opts = useQuery<SqdOptions>("/api/sqd/options");
  const { job } = useJobs();
  const [f, setF] = useState<Form>(DEFAULTS);
  const [planJobId, setPlanJobId] = useState<string | null>(null);
  const [runJobId, setRunJobId] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => {
    setF((p) => ({ ...p, [k]: v }));
    setPlanJobId(null); // any change makes the old plan stale
    setTyped("");
  };

  const planJob = job(planJobId);
  const planId = planJob?.status === "done" ? (planJob.result_hashes[0] ?? null) : null;
  const plan = useQuery<SqdHardwarePlan>(planId ? `/api/sqd/hardware/plans/${planId}` : null);
  const runJob = job(runJobId);
  const planning = planJob && (planJob.status === "queued" || planJob.status === "running");
  const running = runJob && (runJob.status === "queued" || runJob.status === "running");

  useEffect(() => {
    if (runJob?.status === "done" && runJob.result_hashes[0]) onFinished(runJob.result_hashes[0]);
  }, [runJob?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  if (status.loading && !status.data) return null;
  const st = status.data;
  const cli = `miraichem serve --allow-hardware`;

  if (!st?.enabled) {
    return (
      <Panel title="Run on real quantum hardware" aside="switched off">
        <p className="small muted" style={{ maxWidth: "70ch" }}>
          The app cannot use a real quantum computer unless you start it with an explicit opt-in. Restart it like this, then reload
          this page:
        </p>
        <code className="code">{cli}</code>
        <div style={{ marginTop: 8 }}>
          <CopyButton text={cli} />
        </div>
        <p className="small faint" style={{ marginTop: 10, maxWidth: "70ch" }}>
          Even then, nothing is sent until you make a plan, read its estimated QPU time, and type a confirmation. Needs an IBM token
          in <code>.env</code>.
        </p>
      </Panel>
    );
  }

  const mols = opts.data?.molecules ?? [];
  const mol = mols.find((m) => m.id === f.molecule);
  const p = plan.data;
  const canSubmit = !!p && !p.expired && !p.submitted && p.problems.length === 0 && typed.trim() === st.confirm_phrase && !running && !busy;

  const makePlan = async () => {
    setBusy(true);
    setErr(null);
    setTyped("");
    setRunJobId(null);
    try {
      const j = await api.post<Job>("/api/sqd/hardware/plan", {
        molecule: f.molecule,
        backend: f.backend.trim() || null,
        shots: f.shots,
        iterations: f.iterations,
        samples_per_batch: f.samples_per_batch,
        n_reps: f.n_reps,
        seed: f.seed,
      });
      setPlanJobId(j.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!p) return;
    setBusy(true);
    setErr(null);
    try {
      const j = await api.post<Job>("/api/sqd/hardware/run", { plan_id: p.plan_id, confirm: typed });
      setRunJobId(j.id);
      setTyped("");
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const num = (k: keyof Form, step?: string) => (
    <input className="input num" type="number" step={step} value={f[k]} onChange={(e) => set(k, Number(e.target.value) as never)} />
  );

  return (
    <Panel title="Run on real quantum hardware" aside="uses your IBM QPU time">
      <Notice tone="warn">
        Real hardware is <strong>enabled</strong> on this server. A run uses your limited IBM quota. Planning is free; nothing is
        submitted until you type the confirmation. Each run is capped at {st.max_qpu_seconds.toFixed(0)} s of estimated QPU time.
        {!st.token_configured && " No IBM token was found in .env, so planning will fail."}
      </Notice>

      <div className="form" style={{ marginTop: 14 }}>
        <Select
          label="Molecule"
          value={f.molecule}
          onChange={(v) => set("molecule", v)}
          options={mols.map((m) => ({ id: m.id, label: `${molName(m.name)}, ${m.qubits} qubits` }))}
          hint={mol ? `${mol.basis}; ${mol.has_exact ? "checked against the exact energy" : "no exact energy at this size: compared with Hartree-Fock and CCSD"}` : undefined}
        />
        <div className="form__row">
          <Field label="Device" hint="leave empty for the least busy device">
            <input className="input" type="text" placeholder="least busy" value={f.backend} onChange={(e) => set("backend", e.target.value)} />
          </Field>
          <Field label="Shots" hint="more shots cost more QPU time">{num("shots", "1000")}</Field>
        </div>
        <div className="form__row">
          <Field label="Rounds" hint="classical recovery rounds">{num("iterations")}</Field>
          <Field label="Samples per batch" hint="bitstrings per subspace">{num("samples_per_batch", "50")}</Field>
        </div>
        <div className="form__row">
          <Field label="Circuit layers">{num("n_reps")}</Field>
          <Field label="Random seed">{num("seed")}</Field>
        </div>
        <div className="form__submit">
          <button type="button" className="btn" disabled={busy || !!planning || !!running} onClick={makePlan}>
            {planning ? "Planning…" : "Make a plan (uses no QPU time)"}
          </button>
          {planning && <span className="small muted">{planJob?.message}</span>}
        </div>
        {err && <Notice tone="bad">{err}</Notice>}
        {planJob?.status === "failed" && <Notice tone="bad">{planJob.error}</Notice>}
      </div>

      {p && (
        <div style={{ marginTop: 18, display: "grid", gap: 14 }}>
          <div className="tablewrap">
            <table className="table">
              <tbody>
                <tr><td>Device</td><td className="right num">{p.backend}</td></tr>
                <tr><td>Qubits</td><td className="right num">{p.qubits} ({p.qubits_used} used after routing)</td></tr>
                <tr><td>Circuit</td><td className="right num">{p.two_qubit_gates.toLocaleString("en")} two-qubit gates, depth {p.depth.toLocaleString("en")}</td></tr>
                <tr><td>Shots</td><td className="right num">{p.shots.toLocaleString("en")} at about {p.per_shot_us.toFixed(0)} µs each</td></tr>
                <tr className="is-best"><td><strong style={{ fontWeight: 500 }}>Estimated QPU time</strong></td><td className="right num"><strong style={{ fontWeight: 500 }}>about {secs(p.estimated_qpu_s)}</strong></td></tr>
                <tr><td>Queue on the device</td><td className="right num">{p.pending_jobs === null ? "unknown" : `${p.pending_jobs.toLocaleString("en")} jobs ahead`}; wait time is not included</td></tr>
                <tr><td>Classical post-processing</td><td className="right num">about {secs(p.estimated_post_s)}</td></tr>
                <tr><td>QPU time left this period</td><td className="right num">{p.remaining_qpu_s === null ? "unknown" : `${p.remaining_qpu_s.toFixed(0)} s of ${p.budget_s.toFixed(0)} s`}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ maxWidth: "76ch" }}>
            Expected total: about {secs(p.estimated_qpu_s)} on the device, plus the queue (unknown) and about {secs(p.estimated_post_s)} of
            classical work afterwards. The QPU figure is a conservative upper bound; the real bill is shown when the job finishes.
          </p>
          {p.two_qubit_gates > 1000 && (
            <Notice tone="warn">
              This circuit has {p.two_qubit_gates.toLocaleString("en")} two-qubit gates, so most sampled bitstrings will be noise. SQD&apos;s
              recovery step is designed for that, but expect a rough result at this size.
            </Notice>
          )}
          {p.problems.map((m) => (
            <Notice key={m} tone="bad">{m}</Notice>
          ))}
          {p.expired && <Notice tone="bad">This plan has expired. Make a new plan.</Notice>}
          {p.submitted && !running && <Notice tone="warn">This plan was already submitted. Make a new plan to run again.</Notice>}

          {!p.expired && !p.submitted && p.problems.length === 0 && !runJob && (
            <div className="form">
              <Field label={`To submit this to real hardware, type ${st.confirm_phrase}`}>
                <input className="input" type="text" autoComplete="off" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder={st.confirm_phrase} />
              </Field>
              <div className="form__submit">
                <button type="button" className="btn" disabled={!canSubmit} onClick={submit}>
                  Submit to real hardware
                </button>
                <span className="small muted">The plan is valid for {Math.round(st.plan_valid_seconds / 60)} minutes.</span>
              </div>
            </div>
          )}
        </div>
      )}

      {runJob && (
        <div style={{ marginTop: 18, display: "grid", gap: 8 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
            <strong style={{ fontWeight: 400 }}>{runJob.title}</strong>
            <span className={`pill ${runJob.status === "done" ? "pill--ok" : runJob.status === "failed" ? "pill--bad" : ""}`}>{runJob.status}</span>
          </div>
          <p className="small muted num">
            {runJob.message || "Starting…"} · {duration(runJob.elapsed_s)} elapsed
          </p>
          {runJob.details.ibm_job_id && (
            <p className="small muted">
              IBM job <code>{runJob.details.ibm_job_id}</code> is saved. If you close this page or the app, finish it later with{" "}
              <code>miraichem sqd-hardware-fetch {runJob.details.ibm_job_id}</code>.
            </p>
          )}
          {runJob.status === "failed" && <Notice tone="bad">{runJob.error}</Notice>}
        </div>
      )}
    </Panel>
  );
}
