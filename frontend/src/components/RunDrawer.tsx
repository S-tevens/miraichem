import { useEffect, useRef } from "react";
import { qs, useQuery } from "../api";
import { useDrawer, useSource } from "../app/state";
import { deviceName, duration, mha, moleculeName, parts, shortName, when } from "../lib/format";
import { CopyButton, Loading, Notice } from "../ui/ui";
import { CHEM_ACC_MHA, type RunDetail } from "../types";
import { ConvergenceChart } from "./ConvergenceChart";

/** Slide-over with everything saved about one run. Opened from any table row. */
export function RunDrawer() {
  const { hash, source: drawerSource, close } = useDrawer();
  const { source } = useSource();
  const q = useQuery<RunDetail>(hash ? `/api/runs/${hash}${qs({ source: drawerSource ?? source })}` : null);
  const closeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!hash) return;
    closeBtn.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hash, close]);

  if (!hash) return null;
  const row = q.data?.row;
  const run = q.data?.run;
  const p = row ? parts(row) : null;
  const ok = row?.error_mha !== null && row?.error_mha !== undefined && row.error_mha < CHEM_ACC_MHA;

  return (
    <>
      <div className="scrim" onClick={close} aria-hidden="true" />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Run details">
        <div className="drawer__head">
          <div>
            <div className="chips" style={{ marginBottom: 8 }}>
              {row && (
                <>
                  <span className="pill">
                    {moleculeName(row.molecule)} · {row.bond_length} Å
                  </span>
                  <span className="pill">{row.backend === "ideal" ? "Ideal simulator" : deviceName(row.device)}</span>
                  <span className="pill num">seed {row.seed}</span>
                </>
              )}
            </div>
            <h2 className="drawer__title">{row ? shortName(row) : "Run"}</h2>
            <p className="small faint num">{hash}</p>
          </div>
          <button ref={closeBtn} type="button" className="btn btn--ghost btn--small" onClick={close}>
            Close
          </button>
        </div>

        {q.loading && !q.data && <Loading label="Loading run" />}
        {q.error && <Notice tone="bad">{q.error}</Notice>}

        {row && run && p && (
          <>
            {row.status === "failed" ? (
              <Notice tone="bad">This run failed: {row.error_message ?? "no message was saved"}</Notice>
            ) : (
              <div className="stats">
                <div className="stat">
                  <span className="stat__label">Error vs exact</span>
                  <span className="stat__value num">
                    {mha(row.error_mha)} <small>mHa</small>
                  </span>
                  <span className={`pill ${ok ? "pill--ok" : "pill--warn"}`} style={{ justifySelf: "start" }}>
                    {ok ? "within chemical accuracy" : `above ${CHEM_ACC_MHA} mHa`}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat__label">Hartree-Fock error</span>
                  <span className="stat__value num">
                    {mha(row.hf_error_mha)} <small>mHa</small>
                  </span>
                  <span className="stat__note">
                    {row.error_mha !== null && row.hf_error_mha
                      ? row.error_mha < row.hf_error_mha
                        ? `${(row.hf_error_mha / row.error_mha).toFixed(1)}× better than HF`
                        : `${(row.error_mha / row.hf_error_mha).toFixed(1)}× worse than HF`
                      : ""}
                  </span>
                </div>
              </div>
            )}

            <ConvergenceChart
              trace={run.convergence_trace}
              unmitigated={run.convergence_trace_unmitigated}
              eExact={run.e_exact}
              eHf={run.e_hf}
            />

            <dl className="kv">
              <div>
                <dt>VQE energy</dt>
                <dd className="num">{row.e_vqe?.toFixed(6) ?? "n/a"} Ha</dd>
              </div>
              <div>
                <dt>Exact energy</dt>
                <dd className="num">{row.e_exact?.toFixed(6) ?? "n/a"} Ha</dd>
              </div>
              {run.e_vqe_unmitigated !== null && (
                <div>
                  <dt>Before mitigation</dt>
                  <dd className="num">{run.e_vqe_unmitigated.toFixed(6)} Ha</dd>
                </div>
              )}
              <div>
                <dt>Mapping · ansatz</dt>
                <dd>
                  {p.mapping} · {p.ansatz}
                </dd>
              </div>
              <div>
                <dt>Optimizer</dt>
                <dd>
                  {p.optimizer}, {row.maxiter} evaluations
                </dd>
              </div>
              <div>
                <dt>Mitigation</dt>
                <dd>{p.mitigation}</dd>
              </div>
              <div>
                <dt>Qubits · parameters</dt>
                <dd className="num">
                  {row.qubits ?? "n/a"} · {row.params ?? "n/a"}
                </dd>
              </div>
              <div>
                <dt>Two-qubit gates · depth</dt>
                <dd className="num">
                  {row.gates ?? "n/a"} · {row.depth ?? "n/a"}
                </dd>
              </div>
              <div>
                <dt>Total shots</dt>
                <dd className="num">{row.total_shots ? row.total_shots.toLocaleString("en") : "exact (no shots)"}</dd>
              </div>
              <div>
                <dt>Run time</dt>
                <dd className="num">{duration(row.wall_time_s)}</dd>
              </div>
            </dl>

            <div>
              <h3 className="small muted" style={{ marginBottom: 6 }}>
                Reproduce it
              </h3>
              <code className="code">{q.data!.command}</code>
              <div style={{ marginTop: 8 }}>
                <CopyButton text={q.data!.command} />
              </div>
            </div>

            <p className="small faint">
              Saved {when(run.timestamp)}
              {run.git_commit ? ` · commit ${run.git_commit}` : ""} · {Object.keys(run.library_versions).length} library
              versions recorded with the run.
            </p>
          </>
        )}
      </aside>
    </>
  );
}
