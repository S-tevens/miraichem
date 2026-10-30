import { qs, useQuery } from "../api";
import { useJobs, useSource, useDrawer } from "../app/state";
import type { Route } from "../app/router";
import { HalftoneField } from "../components/HalftoneField";
import { RunsTable } from "../components/RunsTable";
import { deviceName, mha, moleculeName, shortName } from "../lib/format";
import { Empty, Loading, Notice, Panel, Stat } from "../ui/ui";
import type { Overview as OverviewData, Study } from "../types";
import "./Overview.css";

export function OverviewPage({ go }: { go: (r: Route) => void }) {
  const { source, sources } = useSource();
  const { finishedCount, active } = useJobs();
  const drawer = useDrawer();
  const ov = useQuery<OverviewData>(`/api/overview${qs({ source, n: finishedCount })}`);
  const reps = useQuery<{ studies: Study[] }>(`/api/replicated${qs({ source, n: finishedCount })}`);

  const featured = (reps.data?.studies ?? [])
    .filter((s) => s.recommendation && s.backend === "noisy" && (s.device === "FakeSherbrooke" || s.device === ""))
    .slice(0, 2);
  const label = sources.find((s) => s.id === source)?.label ?? source;

  return (
    <div className="page">
      <section className="banner" aria-labelledby="banner-h">
        <HalftoneField />
        <div className="banner__shade" />
        <div className="banner__copy">
          <h1 id="banner-h" className="banner__title">
            Find the VQE setup you can trust on a noisy quantum computer
          </h1>
          <p className="banner__sub">
            Benchmark circuits, optimizers and error mitigation for small molecules, and get a
            recommendation backed by repeated runs.
          </p>
          <div className="banner__actions">
            <button type="button" className="btn" onClick={() => go("new-run")}>
              Start a run
            </button>
            <button type="button" className="btn btn--ghost" onClick={() => go("leaderboard")}>
              Open the leaderboard
            </button>
          </div>
        </div>
      </section>

      {ov.error && <Notice tone="bad">{ov.error}</Notice>}
      {ov.loading && !ov.data && <Loading />}

      {ov.data && ov.data.n_runs === 0 && (
        <Panel>
          <Empty
            title={`No runs in “${label}” yet`}
            art="stagger"
            action={
              <button type="button" className="btn" onClick={() => go("new-run")}>
                Run your first configuration
              </button>
            }
          >
            Start a single run, or launch one of the prepared sweeps. You can also switch to the published
            results in the sidebar.
          </Empty>
        </Panel>
      )}

      {ov.data && ov.data.n_runs > 0 && (
        <>
          <div className="stats">
            <Stat label="Runs" value={ov.data.n_ok.toLocaleString("en")} note={ov.data.n_failed ? `${ov.data.n_failed} failed` : label} />
            <Stat
              label="Molecules"
              value={Object.keys(ov.data.by_molecule).map(moleculeName).join(" · ")}
              note={Object.entries(ov.data.by_molecule)
                .map(([m, b]) => `${moleculeName(m)}: ${Object.entries(b).map(([k, n]) => `${n} ${k}`).join(", ")}`)
                .join(" · ")}
            />
            <Stat label="Real hardware results" value={ov.data.hardware_results} note="saved job records" />
            <Stat label="Dissociation curves" value={ov.data.curves} />
            {active > 0 && <Stat label="Running now" value={active} note="see Jobs" />}
          </div>

          {featured.length > 0 && (
            <div className="grid grid--2">
              {featured.map((s) => {
                const rec = s.recommendation!;
                const c = s.configs.find((x) => x.label === rec.label);
                if (!c) return null;
                return (
                  <Panel
                    key={s.id}
                    title={`Recommended for ${moleculeName(s.molecule)}`}
                    aside={`${s.backend === "noisy" ? deviceName(s.device) : "ideal"} · ${s.bond_length} Å`}
                  >
                    <div className="rec">
                      <p className="rec__name">{shortName(c)}</p>
                      <div className="chips">
                        <span className={`pill ${rec.reliable ? "pill--ok" : "pill--warn"}`}>
                          {rec.reliable ? "reliably chemically accurate" : "not reliably chemically accurate"}
                        </span>
                        <span className="pill num">
                          {mha(c.mean)} ± {mha(c.std)} mHa
                        </span>
                        <span className="pill num">
                          {c.n_chem}/{c.n_seeds} seeds within 1.6 mHa
                        </span>
                      </div>
                      <div className="chips">
                        <button type="button" className="btn btn--ghost btn--small" onClick={() => go("reliability")}>
                          See the evidence
                        </button>
                        <a
                          className="btn btn--ghost btn--small"
                          href={`#/trust?kind=vqe&molecule=${encodeURIComponent(s.molecule)}&backend=${s.backend}&device=${encodeURIComponent(s.device)}`}
                        >
                          Trust report
                        </a>
                      </div>
                    </div>
                  </Panel>
                );
              })}
            </div>
          )}

          <Panel title="Recent runs" aside={`from ${label}`}>
            <RunsTable rows={ov.data.recent} onOpen={(h) => drawer.open(h, source)} compact />
          </Panel>
        </>
      )}
    </div>
  );
}
