import { useEffect, useMemo, useState } from "react";
import { qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import type { Route } from "../app/router";
import { deviceName, moleculeName } from "../lib/format";
import { Empty, Loading, Notice, PageHead, Panel, Select } from "../ui/ui";
import type { TrustReport, TrustStatus, TrustSubject } from "../types";
import "./Trust.css";

const MARK: Record<TrustStatus, string> = { pass: "✓", warn: "!", fail: "✕", none: "–" };
const MARK_NAME: Record<TrustStatus, string> = { pass: "passes", warn: "warning", fail: "fails", none: "no data" };

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const sqdName = (n: string) => {
  const [head, ...rest] = n.split("_");
  const f = head.replace(/\d/g, (d) => SUB[Number(d)]);
  return rest.length ? `${f} (${rest.join(" ")})` : f;
};

const subjectId = (s: TrustSubject) => [s.kind, s.molecule, s.backend, s.device].join("|");
const subjectLabel = (s: TrustSubject) =>
  s.kind === "sqd"
    ? `${sqdName(s.molecule)}, SQD`
    : `${moleculeName(s.molecule)}, VQE, ${s.backend}${s.device ? `, ${deviceName(s.device)}` : ""}`;

/** The query part of the hash (`#/trust?kind=vqe&molecule=H2`), kept live as the hash changes. */
function useHashQuery(): URLSearchParams {
  const read = () => new URLSearchParams(window.location.hash.split("?")[1] ?? "");
  const [q, setQ] = useState(read);
  useEffect(() => {
    const on = () => setQ(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return q;
}

export function TrustPage({ go }: { go: (r: Route) => void }) {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const link = useHashQuery();
  const index = useQuery<{ subjects: TrustSubject[] }>(`/api/trust/index${qs({ source, n: finishedCount })}`);
  const subjects = useMemo(() => index.data?.subjects ?? [], [index.data]);
  const [picked, setPicked] = useState<string | null>(null);

  // A link from another page names a subject by some of its fields; the first match is shown.
  const fromLink = useMemo(() => {
    const want = { kind: link.get("kind"), molecule: link.get("molecule"), backend: link.get("backend"), device: link.get("device") };
    if (!want.kind || !want.molecule) return null;
    return (
      subjects.find(
        (s) =>
          s.kind === want.kind &&
          s.molecule.toLowerCase() === want.molecule!.toLowerCase() &&
          (!want.backend || s.backend === want.backend) &&
          (want.device === null || s.device === want.device),
      ) ?? null
    );
  }, [link, subjects]);

  useEffect(() => {
    if (fromLink) setPicked(subjectId(fromLink));
  }, [fromLink]);

  const current = subjects.find((s) => subjectId(s) === picked) ?? subjects.find((s) => s.kind === "vqe") ?? subjects[0];
  const report = useQuery<TrustReport>(
    current
      ? `/api/trust${qs({ kind: current.kind, molecule: current.molecule, backend: current.backend, device: current.device, source, n: finishedCount })}`
      : null,
  );

  if (index.loading && !index.data) return <div className="page"><Loading /></div>;
  if (index.error) return <div className="page"><Notice tone="bad">{index.error}</Notice></div>;

  return (
    <div className="page">
      <PageHead
        title="Trust report"
        sub="For each recommendation, what was checked and how much to rely on it. The verdict follows fixed rules, listed below, so nothing about it is hidden."
        tools={
          subjects.length > 1 ? (
            <Select
              label="Recommendation"
              value={current ? subjectId(current) : ""}
              onChange={setPicked}
              options={subjects.map((s) => ({ id: subjectId(s), label: subjectLabel(s) }))}
            />
          ) : undefined
        }
      />

      {!current && (
        <Panel>
          <Empty title="No recommendations in this source yet" art="scatter">
            A report needs a seed study with a recommendation, or an SQD seed study. Run a sweep with several seeds first.
          </Empty>
        </Panel>
      )}

      {report.loading && !report.data && current && <Loading label="Collecting the evidence" />}
      {report.error && <Notice tone="bad">{report.error}</Notice>}

      {report.data && current && (
        <>
          <section className={`trust__verdict trust__verdict--${report.data.verdict}`} aria-label="Verdict">
            <span className="small muted">{subjectLabel(current)}</span>
            <span className="trust__label">{report.data.verdict_text}</span>
            <p className="small muted" style={{ maxWidth: "76ch" }}>{report.data.sentence}</p>
            {report.data.subject.label && <p className="trust__setup">Recommended setup: {report.data.subject.label}</p>}
            <div className="no-print" style={{ marginTop: 6 }}>
              <button type="button" className="btn btn--ghost btn--small" onClick={() => window.print()}>
                Print this report
              </button>
            </div>
          </section>

          {report.data.evidence.length > 0 && (
            <Panel title="Evidence" aside="what was checked">
              <ul className="trust__list">
                {report.data.evidence.map((e) => (
                  <li key={e.id} className="trust__row">
                    <span className={`trust__mark trust__mark--${e.status}`} role="img" aria-label={MARK_NAME[e.status]}>{MARK[e.status]}</span>
                    <div>
                      <div className="trust__title">{e.title}</div>
                      <div className="trust__detail">{e.detail}</div>
                    </div>
                    {e.route && (
                      <button type="button" className="btn btn--ghost btn--small no-print" onClick={() => go(e.route as Route)}>
                        See it
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title="How the verdict is decided">
            <ol className="trust__rules small muted">
              {report.data.rules.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ol>
            <p className="small faint" style={{ marginTop: 10, maxWidth: "78ch" }}>
              A trusted verdict is about the setup&apos;s behaviour on small molecules we can check exactly. It is not a claim that the method helps
              on molecules too big to solve classically, and no result here is a quantum advantage.
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}
