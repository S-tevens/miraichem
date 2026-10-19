import { useState } from "react";
import { ApiError, useQuery } from "../api";
import type { Route } from "../app/router";
import { useJobs } from "../app/state";
import { JobCard } from "../components/JobCard";
import { DEMO } from "../lib/demo";
import { deviceName, moleculeName } from "../lib/format";
import { Empty, Loading, Notice, PageHead, Panel, Progress } from "../ui/ui";
import type { Job, SweepInfo } from "../types";
import "./Sweeps.css";

export function SweepsPage({ go }: { go: (r: Route) => void }) {
  if (DEMO) {
    return (
      <div className="page">
        <PageHead title="Sweeps" />
        <Panel>
          <Empty title="Sweeps run in the live app" art="stagger">
            This offline demo shows saved results only. Start the app with <code>miraichem serve</code> to launch one of the
            prepared sweeps with live progress.
          </Empty>
        </Panel>
      </div>
    );
  }
  return <SweepsLive go={go} />;
}

function SweepsLive({ go }: { go: (r: Route) => void }) {
  const { startSweep, jobs, finishedCount } = useJobs();
  const q = useQuery<{ sweeps: SweepInfo[]; busy: boolean }>(`/api/sweeps?n=${finishedCount}`);
  const [open, setOpen] = useState<string | null>(null);
  const [workers, setWorkers] = useState(4);
  const [force, setForce] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const sweepJobs = jobs.filter((j) => j.kind === "sweep");
  const runningSweep = sweepJobs.find((j) => j.status === "queued" || j.status === "running");

  const start = async (name: string) => {
    setBusy(true);
    setError(null);
    try {
      await startSweep(name, workers, force);
      setOpen(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <PageHead
        title="Sweeps"
        sub="A sweep runs every combination of molecule, circuit, optimizer, mitigation, device and seed in a configuration file. Finished runs are cached, so a sweep can be stopped and resumed."
      />

      {runningSweep && (
        <Panel title="Running now">
          <JobCard job={runningSweep} onOpenResults={() => go("leaderboard")} />
        </Panel>
      )}

      {q.loading && !q.data && <Loading />}
      {q.error && <Notice tone="bad">{q.error}</Notice>}
      {error && <Notice tone="bad">{error}</Notice>}

      {q.data && (
        <div className="grid sweeps">
          {q.data.sweeps.map((s) => {
            if (s.error) {
              return (
                <Panel key={s.name} title={s.name}>
                  <Notice tone="bad">{s.error}</Notice>
                </Panel>
              );
            }
            const remaining = (s.n_runs ?? 0) - (s.n_cached ?? 0);
            const isOpen = open === s.name;
            return (
              <Panel key={s.name} className="sweep">
                <div className="sweep__head">
                  <h2 className="panel__title">{s.name}</h2>
                  <div className="chips">
                    <span className="pill">{moleculeName(s.molecule ?? "")}</span>
                    {(s.backends ?? []).map((b) => (
                      <span key={b} className="pill">{b === "ideal" ? "ideal" : "noisy"}</span>
                    ))}
                    {(s.seeds ?? 1) > 1 && <span className="pill num">{s.seeds} seeds</span>}
                  </div>
                </div>
                <p className="small muted sweep__desc">{s.description || "No description."}</p>
                {(s.devices?.length ?? 0) > 0 && (
                  <p className="small faint">Devices: {s.devices!.map(deviceName).join(", ")}</p>
                )}
                <div className="sweep__prog">
                  <Progress value={s.n_cached ?? 0} total={s.n_runs ?? 0} tone={remaining === 0 ? "ok" : undefined} />
                  <span className="small muted num">
                    {s.n_cached} of {s.n_runs} runs already saved
                    {remaining > 0 ? ` · ${remaining} to run` : " · complete"}
                  </span>
                </div>

                {isOpen ? (
                  <div className="sweep__start">
                    <label className="field">
                      <span className="field__label">Parallel workers</span>
                      <select className="select" value={workers} onChange={(e) => setWorkers(Number(e.target.value))}>
                        {[1, 2, 4, 8].map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                      <span className="field__hint">more workers finish sooner but use more of your CPU</span>
                    </label>
                    <label className="check">
                      <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                      <span>Run everything again, including saved runs</span>
                    </label>
                    {remaining > 150 && (
                      <Notice tone="warn">This will run {force ? s.n_runs : remaining} runs, which can take a long time. You can leave the page; progress is kept.</Notice>
                    )}
                    <div className="chips">
                      <button type="button" className="btn" disabled={busy || !!runningSweep} onClick={() => start(s.name)}>
                        {busy ? "Starting…" : `Start ${force ? s.n_runs : remaining} runs`}
                      </button>
                      <button type="button" className="btn btn--ghost" onClick={() => setOpen(null)}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    disabled={!!runningSweep || (remaining === 0 && !force)}
                    onClick={() => setOpen(s.name)}
                    title={runningSweep ? "Another sweep is running" : undefined}
                  >
                    {remaining === 0 ? "Already complete" : "Prepare to run"}
                  </button>
                )}
              </Panel>
            );
          })}
        </div>
      )}

      {sweepJobs.filter((j: Job) => j !== runningSweep).length > 0 && (
        <Panel title="Earlier sweeps">
          <div className="stack">
            {sweepJobs
              .filter((j) => j !== runningSweep)
              .slice(0, 5)
              .map((j) => (
                <JobCard key={j.id} job={j} onOpenResults={() => go("leaderboard")} />
              ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
