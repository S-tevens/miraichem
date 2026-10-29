import { useEffect, useRef, type ReactNode } from "react";
import { DEMO } from "../lib/demo";
import { Icon } from "../ui/ui";
import { useJobs, useSource } from "./state";
import { ROUTES, type Route } from "./router";

const GROUPS: { title: string; items: { id: Route; label: string }[] }[] = [
  {
    title: "Explore",
    items: [
      { id: "overview", label: "Overview" },
      { id: "leaderboard", label: "Leaderboard" },
      { id: "reliability", label: "Seed reliability" },
      { id: "devices", label: "Devices" },
    ],
  },
  {
    title: "Analyze",
    items: [
      { id: "curves", label: "Dissociation curves" },
      { id: "hardware", label: "Real hardware" },
      { id: "predictor", label: "Error predictor" },
      { id: "sqd", label: "SQD, larger molecules" },
    ],
  },
  {
    title: "Run",
    items: [
      { id: "new-run", label: "New run" },
      { id: "sweeps", label: "Sweeps" },
      { id: "jobs", label: "Jobs" },
    ],
  },
  { title: "Help", items: [{ id: "about", label: "How it works" }] },
];

/** The mark: a small radial dot matrix, in the reference's halftone language. */
function Mark() {
  const dots: { x: number; y: number; r: number; o: number }[] = [];
  for (let j = -2; j <= 2; j++)
    for (let i = -2; i <= 2; i++) {
      const d = Math.hypot(i, j);
      if (d > 2.3) continue;
      dots.push({ x: 15 + i * 5.6, y: 15 + j * 5.6, r: 2.3 - d * 0.62, o: 1 - d * 0.22 });
    }
  return (
    <svg viewBox="0 0 30 30" className="brand__mark" aria-hidden="true">
      {dots.map((d, k) => (
        <circle key={k} cx={d.x} cy={d.y} r={Math.max(d.r, 0.6)} fill="#a9d6d1" opacity={d.o} />
      ))}
    </svg>
  );
}

export function Shell({
  route,
  go,
  children,
}: {
  route: Route;
  go: (r: Route) => void;
  children: ReactNode;
}) {
  const { source, sources, setSource, connected } = useSource();
  const { active } = useJobs();
  const mainRef = useRef<HTMLElement>(null);

  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [route]);

  return (
    <div className="app">
      <aside className="side">
        <a className="brand" href="#/overview" aria-label="MiraiChem, home">
          <Mark />
          <span className="brand__name">MiraiChem</span>
        </a>

        <nav className="nav" aria-label="Main">
          {GROUPS.map((g) => (
            <div key={g.title} style={{ display: "contents" }}>
              <span className="nav__group">{g.title}</span>
              {g.items.map((it) => (
                <a
                  key={it.id}
                  className="nav__item"
                  href={`#/${it.id}`}
                  aria-current={route === it.id ? "page" : undefined}
                  onClick={(e) => {
                    e.preventDefault();
                    go(it.id);
                  }}
                >
                  <Icon name={it.id} />
                  {it.label}
                  {it.id === "jobs" && active > 0 && (
                    <span className="nav__badge num" aria-label={`${active} running`}>
                      {active}
                    </span>
                  )}
                </a>
              ))}
            </div>
          ))}
        </nav>

        <div className="side__foot">
          {DEMO ? (
            <div className="source">
              <span className="source__label">Offline demo</span>
              <span className="small muted">Published results, read-only.</span>
            </div>
          ) : (
          <div className="source">
              <span className="source__label">Data source</span>
              <div className="seg" role="group" aria-label="Data source" style={{ width: "100%" }}>
                {sources.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    aria-pressed={s.id === source}
                    style={{ flex: 1 }}
                    title={`${s.path}: ${s.n_runs} runs`}
                    onClick={() => setSource(s.id)}
                  >
                    {s.id === "local" ? "My runs" : "Published"}
                  </button>
                ))}
              </div>
              <span className="small faint num">
                {sources.find((s) => s.id === source)?.n_runs.toLocaleString("en") ?? 0} runs
              </span>
            </div>
          )}
          <div className="status" role="status">
            <span
              className={`status__dot ${connected ? "status__dot--ok" : connected === false ? "status__dot--bad" : ""}`}
            />
            {DEMO ? "Static data, no server" : connected === null ? "Connecting…" : connected ? "Server connected" : "Server not reachable"}
          </div>
        </div>
      </aside>

      <main className="main" ref={mainRef}>
        {children}
      </main>
    </div>
  );
}

export { ROUTES };
