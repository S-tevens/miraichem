import { useEffect, useState } from "react";
import { ApiError, qs, useQuery } from "../api";
import { useDrawer, useJobs } from "../app/state";
import { ConvergenceChart } from "../components/ConvergenceChart";
import { duration, mha, moleculeName } from "../lib/format";
import { Field, Loading, Notice, PageHead, Panel, Progress, Select, Seg, Stat } from "../ui/ui";
import { CHEM_ACC_MHA, type Job, type Options, type Reference, type RunDetail, type RunRequest } from "../types";
import "./NewRun.css";

const DEFAULTS: RunRequest = {
  molecule: "h2",
  bond_length: 0.735,
  mapping: "parity",
  ansatz: "uccsd",
  ansatz_reps: 1,
  optimizer: "lbfgsb",
  maxiter: 150,
  backend: "ideal",
  fake_backend_name: "FakeSherbrooke",
  shots: 4096,
  mitigation: "none",
  seed: 42,
  force: false,
};

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

/** What the backend will refuse, explained before the user presses Run. */
function problems(f: RunRequest): string[] {
  const out: string[] = [];
  if (f.optimizer === "lbfgsb" && f.backend !== "ideal")
    out.push("L-BFGS-B needs exact values and gradients, so it only works on the ideal simulator.");
  if (f.mitigation !== "none" && f.backend === "ideal")
    out.push("Error mitigation only makes sense on the noisy simulator.");
  if (f.mitigation.includes("zne") && f.backend === "noisy" && f.maxiter > 400)
    out.push("ZNE triples the circuits per evaluation, so a large budget will take a long time.");
  return out;
}

function estimate(f: RunRequest, qubitsJw: number | undefined): string {
  if (f.backend === "ideal") return "usually a few seconds";
  const heavy = (qubitsJw ?? 4) > 4 && f.mapping === "jordan_wigner";
  const base = f.maxiter * (f.mitigation.includes("zne") ? 0.12 : 0.07) * (f.ansatz === "uccsd" ? 1.5 : 1) * (heavy ? 4 : 1);
  return base < 20 ? "under half a minute" : `roughly ${duration(base)} or more`;
}

export function NewRunPage() {
  const { startRun, job, finishedCount } = useJobs();
  const drawer = useDrawer();
  const opts = useQuery<Options>("/api/options");
  const [f, setF] = useState<RunRequest>(DEFAULTS);
  const set = <K extends keyof RunRequest>(k: K, v: RunRequest[K]) => setF((p) => ({ ...p, [k]: v }));
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const mol = opts.data?.molecules.find((m) => m.id === f.molecule);
  const bond = useDebounced(f.bond_length, 350);
  const ref = useQuery<Reference>(f.molecule ? `/api/reference${qs({ molecule: f.molecule, bond_length: bond })}` : null);

  // keep the form valid as the user changes things
  const setBackend = (b: "ideal" | "noisy") =>
    setF((p) => ({
      ...p,
      backend: b,
      mitigation: b === "ideal" ? "none" : p.mitigation,
      optimizer: b === "noisy" && p.optimizer === "lbfgsb" ? "spsa_clip" : p.optimizer,
    }));

  const issues = problems(f);
  const blocking = issues.filter((i) => !i.startsWith("ZNE triples"));
  const current: Job | undefined = job(jobId);
  const running = current && (current.status === "queued" || current.status === "running");

  const result = useQuery<RunDetail>(
    current?.status === "done" && current.result_hashes[0]
      ? `/api/runs/${current.result_hashes[0]}${qs({ source: "local", n: finishedCount })}`
      : null,
  );

  const submit = async () => {
    setBusy(true);
    setSubmitError(null);
    try {
      const j = await startRun(f);
      setJobId(j.id);
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (opts.loading && !opts.data) return <div className="page"><Loading /></div>;
  if (!opts.data) return <div className="page"><Notice tone="bad">{opts.error ?? "Could not load options."}</Notice></div>;
  const o = opts.data;
  const showLayers = f.ansatz !== "uccsd";
  const noisy = f.backend === "noisy";

  return (
    <div className="page">
      <PageHead
        title="New run"
        sub="Configure one VQE run and start it on a simulator. It is saved with your runs, so it appears in the leaderboard and can be compared with everything else."
      />

      <div className="grid grid--side">
        <Panel title="Configuration">
          <div className="form">
            <Select
              label="Molecule"
              value={f.molecule}
              onChange={(v) => {
                const m = o.molecules.find((x) => x.id === v);
                setF((p) => ({ ...p, molecule: v, bond_length: m?.equilibrium ?? p.bond_length }));
              }}
              options={o.molecules.map((m) => ({ id: m.id, label: moleculeName(m.name) }))}
              hint={mol ? `${mol.basis}${mol.active_electrons ? `, ${mol.active_electrons} active electrons` : ""}` : undefined}
            />

            <Field
              label={`Bond length: ${f.bond_length} Å`}
              hint={mol ? `Equilibrium is ${mol.equilibrium} Å. Stretched bonds are where Hartree-Fock fails.` : undefined}
            >
              <input
                className="range"
                type="range"
                min={mol ? Math.min(...mol.bond_lengths) : 0.3}
                max={mol ? Math.max(...mol.bond_lengths) : 2.5}
                step={0.005}
                value={f.bond_length}
                onChange={(e) => set("bond_length", Number(e.target.value))}
              />
            </Field>

            <Field label="Qubit mapping" hint={o.mappings.find((m) => m.id === f.mapping)?.hint}>
              <Seg label="Mapping" value={f.mapping} onChange={(v) => set("mapping", v)} options={o.mappings} />
            </Field>

            <div className="form__row">
              <Select
                label="Circuit (ansatz)"
                value={f.ansatz}
                onChange={(v) => set("ansatz", v)}
                options={o.ansatzes}
                hint={o.ansatzes.find((a) => a.id === f.ansatz)?.hint}
              />
              {showLayers && (
                <Field label="Layers" hint="more layers express more, and add noise">
                  <input
                    className="input"
                    type="number"
                    min={1}
                    max={6}
                    value={f.ansatz_reps}
                    onChange={(e) => set("ansatz_reps", Math.max(1, Math.min(6, Number(e.target.value) || 1)))}
                  />
                </Field>
              )}
            </div>

            <div className="form__row">
              <Select
                label="Optimizer"
                value={f.optimizer}
                onChange={(v) => set("optimizer", v)}
                options={o.optimizers.map((x) => ({ ...x, label: x.id === "lbfgsb" && noisy ? `${x.label} (ideal only)` : x.label }))}
                hint={o.optimizers.find((x) => x.id === f.optimizer)?.hint}
              />
              <Field label="Evaluation budget" hint="energy estimates the optimizer may use">
                <input
                  className="input num"
                  type="number"
                  min={10}
                  max={5000}
                  step={10}
                  value={f.maxiter}
                  onChange={(e) => set("maxiter", Math.max(1, Number(e.target.value) || 1))}
                />
              </Field>
            </div>

            <Field label="Backend" hint={o.backends.find((b) => b.id === f.backend)?.hint}>
              <Seg label="Backend" value={f.backend} onChange={setBackend} options={o.backends as { id: "ideal" | "noisy"; label: string }[]} />
            </Field>

            {noisy && (
              <>
                <div className="form__row">
                  <Select label="IBM device model" value={f.fake_backend_name} onChange={(v) => set("fake_backend_name", v)} options={o.devices} />
                  <Select
                    label="Shots per circuit"
                    value={String(f.shots)}
                    onChange={(v) => set("shots", Number(v))}
                    options={[1024, 4096, 16384].map((n) => ({ id: String(n), label: n.toLocaleString("en") }))}
                  />
                </div>
                <Select label="Error mitigation" value={f.mitigation} onChange={(v) => set("mitigation", v)} options={o.mitigations} />
              </>
            )}

            <div className="form__row">
              <Field label="Random seed" hint="changes the start point and shot noise">
                <input className="input num" type="number" value={f.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)} />
              </Field>
              <label className="check">
                <input type="checkbox" checked={f.force} onChange={(e) => set("force", e.target.checked)} />
                <span>
                  Run again even if this exact configuration was saved
                </span>
              </label>
            </div>

            {issues.map((i) => (
              <Notice key={i} tone={i.startsWith("ZNE") ? "warn" : "bad"}>{i}</Notice>
            ))}
            {submitError && <Notice tone="bad">{submitError}</Notice>}

            <div className="form__submit">
              <button type="button" className="btn" disabled={busy || !!running || blocking.length > 0} onClick={submit}>
                {running ? "Running…" : "Run on simulator"}
              </button>
              <span className="small muted">Expected: {estimate(f, ref.data?.n_qubits_jw)}.</span>
            </div>
            <p className="small faint">
              Real quantum hardware is not started from here. It needs an explicit opt-in, a cost estimate and
              your confirmation, so it stays in the command line (<code>miraichem hardware</code>).
            </p>
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Classical reference" aside={mol ? `${moleculeName(mol.name)} at ${bond} Å` : undefined}>
            {ref.loading && !ref.data && <Loading label="Computing the reference" />}
            {ref.error && <Notice tone="bad">{ref.error}</Notice>}
            {ref.data && (
              <div className="stats">
                <Stat label="Hartree-Fock energy" value={ref.data.e_hf.toFixed(5)} unit="Ha" note="the mean-field answer" />
                <Stat label={`Exact energy (${ref.data.exact_method})`} value={ref.data.e_exact.toFixed(5)} unit="Ha" note="what VQE is scored against" />
                <Stat label="Hartree-Fock error" value={mha(ref.data.hf_error_mha)} unit="mHa" note={`target: below ${CHEM_ACC_MHA} mHa`} />
              </div>
            )}
          </Panel>

          <Panel title="Result" aside={current ? `${current.title}` : undefined}>
            {!current && (
              <p className="small muted">
                Start a run to see it here. Progress is live, and the finished run opens with its convergence curve.
              </p>
            )}
            {current && (
              <div className="stack">
                <div className="jobline">
                  <span className={`pill ${current.status === "done" ? "pill--ok" : current.status === "failed" ? "pill--bad" : ""}`}>
                    {current.status}
                  </span>
                  <span className="small muted num">{duration(current.elapsed_s)}</span>
                  {running && <div className="spinner" />}
                </div>
                {running && <Progress value={0} total={1} />}
                {current.message && current.status !== "failed" && <p className="small muted">{current.message}</p>}
                {current.status === "failed" && <Notice tone="bad">{current.error ?? "The run failed."}</Notice>}
                {current.status === "done" && result.data && (
                  <>
                    <div className="stats">
                      <Stat label="Error vs exact" value={mha(result.data.row.error_mha)} unit="mHa" note={(result.data.row.error_mha ?? 99) < CHEM_ACC_MHA ? "within chemical accuracy" : `above ${CHEM_ACC_MHA} mHa`} />
                      <Stat label="Energies used" value={result.data.row.evals ?? "n/a"} note={`${duration(result.data.row.wall_time_s)} wall time`} />
                      <Stat label="Two-qubit gates" value={result.data.row.gates ?? "n/a"} note={`${result.data.row.qubits ?? "?"} qubits`} />
                    </div>
                    <ConvergenceChart
                      trace={result.data.run.convergence_trace}
                      unmitigated={result.data.run.convergence_trace_unmitigated}
                      eExact={result.data.run.e_exact}
                      eHf={result.data.run.e_hf}
                    />
                    <button type="button" className="btn btn--ghost btn--small" onClick={() => drawer.open(result.data!.row.hash, "local")}>
                      Open full details
                    </button>
                  </>
                )}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
