import { useEffect, useMemo, useState } from "react";
import { qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import { LihPanel } from "../components/LihPanel";
import { SeedStrip } from "../components/SeedStrip";
import { deviceName, mha, moleculeName, parts, pct, shortName } from "../lib/format";
import { log } from "../lib/scale";
import { Empty, Field, Loading, Notice, PageHead, Panel, Select, Seg } from "../ui/ui";
import { CHEM_ACC_MHA, type Config, type Recommendation, type Study } from "../types";
import "./Reliability.css";

const SHOW = 12;

function MiniBar({ c, max }: { c: Config; max: number }) {
  const x = log([0.1, max], [0, 100]);
  const lo = Math.max(0.1, c.mean - c.std);
  return (
    <span className="mini" aria-hidden="true">
      <span className="mini__spread" style={{ left: `${x(lo)}%`, width: `${x(c.mean + c.std) - x(lo)}%` }} />
      <span className="mini__dot" style={{ left: `${x(c.mean)}%` }} />
    </span>
  );
}

export function ReliabilityPage() {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const q = useQuery<{ studies: Study[] }>(`/api/replicated${qs({ source, n: finishedCount })}`);
  const studies = q.data?.studies ?? [];

  const molecules = useMemo(() => [...new Set(studies.map((s) => s.molecule))], [studies]);
  const [molecule, setMolecule] = useState("");
  const [backend, setBackend] = useState<"noisy" | "ideal">("noisy");
  const [device, setDevice] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [adv, setAdv] = useState({ reliability: 50, alpha: 5, margin: 1.5 });

  useEffect(() => {
    if (!molecule && molecules.length) setMolecule(molecules.includes("H2") ? "H2" : molecules[0]);
  }, [molecules, molecule]);
  const backends = (["noisy", "ideal"] as const).filter((b) => studies.some((s) => s.molecule === molecule && s.backend === b));
  useEffect(() => {
    if (backends.length && !backends.includes(backend)) setBackend(backends[0]);
  }, [backends, backend]);

  const candidates = studies.filter((s) => s.molecule === molecule && s.backend === backend);
  const study = candidates.find((s) => s.device === device) ?? candidates.find((s) => s.device === "FakeSherbrooke") ?? candidates[0];
  useEffect(() => {
    setSelected(null);
    setShowAll(false);
  }, [study?.id]);

  // the recommendation again with the user's own thresholds
  const recPath = study
    ? `/api/replicated/recommendation${qs({
        molecule: study.molecule,
        backend: study.backend,
        device: study.device,
        reliability: adv.reliability / 100,
        alpha: adv.alpha / 100,
        margin_factor: adv.margin,
        source,
        n: finishedCount,
      })}`
    : null;
  const recQ = useQuery<Recommendation>(recPath);
  const rec = recQ.data ?? study?.recommendation ?? null;

  if (q.loading && !q.data) return <div className="page"><Loading /></div>;
  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;
  if (!study) {
    return (
      <div className="page">
        <PageHead title="Seed reliability" />
        <Panel>
          <Empty title="No seed-replicated results in this source" art="scatter">
            A single run of a noisy computation can be lucky. Run a sweep that uses several seeds (for example
            <code> h2_replication</code>) to see mean error, spread and the share of seeds that reach chemical accuracy.
          </Empty>
        </Panel>
      </div>
    );
  }

  const current = study.configs.find((c) => c.label === selected) ?? study.configs.find((c) => c.label === rec?.label) ?? study.configs[0];
  const list = showAll ? study.configs : study.configs.slice(0, SHOW);
  const maxMean = Math.max(...study.configs.map((c) => c.mean + c.std), 10);
  const p = parts(current);
  const isRec = rec?.label === current.label;
  const recConfig = rec ? study.configs.find((c) => c.label === rec.label) : undefined;

  return (
    <div className="page">
      <PageHead
        title="Seed reliability"
        sub="A single run of a noisy computation can be lucky. Each configuration here was repeated with different random seeds, so you can see its typical error and how much it varies."
        tools={
          <>
            <Select label="Molecule" value={molecule} onChange={setMolecule} options={molecules.map((m) => ({ id: m, label: moleculeName(m) }))} />
            <Field label="Backend">
              <Seg label="Backend" value={backend} onChange={setBackend} options={backends.map((b) => ({ id: b, label: b === "noisy" ? "Noisy" : "Ideal" }))} />
            </Field>
            {candidates.length > 1 && (
              <Select
                label="IBM device model"
                value={study.device}
                onChange={setDevice}
                options={candidates.map((s) => ({ id: s.device, label: `${deviceName(s.device)} (${s.configs.length})` }))}
              />
            )}
          </>
        }
      />

      {rec && (
        <Panel
          title="Recommendation from all seeds"
          aside={
            <span className={`pill ${rec.reliable ? "pill--ok" : "pill--warn"}`}>
              {rec.reliable ? "reliably chemically accurate" : "not reliably chemically accurate"}
            </span>
          }
        >
          <div className="grid grid--side">
            <div className="rec2">
              {recConfig && <h3 className="rec2__name">{shortName(recConfig)}</h3>}
              <p className="rec2__text">{rec.justification}</p>
            </div>
            <div className="adv">
              <h4 className="small muted">How strict to be</h4>
              <label className="weight">
                <span className="weight__row"><span>Required reliability</span><span className="num muted">{adv.reliability}% of seeds</span></span>
                <input className="range" type="range" min={10} max={100} step={5} value={adv.reliability} onChange={(e) => setAdv((a) => ({ ...a, reliability: Number(e.target.value) }))} aria-label="Required reliability" />
                <span className="field__hint">share of seeds that must reach {CHEM_ACC_MHA} mHa</span>
              </label>
              <label className="weight">
                <span className="weight__row"><span>Significance level</span><span className="num muted">{adv.alpha}%</span></span>
                <input className="range" type="range" min={1} max={20} step={1} value={adv.alpha} onChange={(e) => setAdv((a) => ({ ...a, alpha: Number(e.target.value) }))} aria-label="Significance level" />
                <span className="field__hint">how sure we must be that a cheaper option is worse before rejecting it</span>
              </label>
              <label className="weight">
                <span className="weight__row"><span>Practical margin</span><span className="num muted">{adv.margin.toFixed(1)}×</span></span>
                <input className="range" type="range" min={1} max={3} step={0.1} value={adv.margin} onChange={(e) => setAdv((a) => ({ ...a, margin: Number(e.target.value) }))} aria-label="Practical margin" />
                <span className="field__hint">how much worse than the best mean still counts as tied</span>
              </label>
            </div>
          </div>
        </Panel>
      )}

      <div className="grid grid--side">
        <Panel title="Configurations" aside={`${list.length} of ${study.configs.length}, best mean error first`}>
          <ul className="items">
            {list.map((c, i) => {
              const cp = parts(c);
              const active = c.label === current.label;
              return (
                <li key={c.label}>
                  <button type="button" className={`item ${active ? "is-active" : ""}`} aria-pressed={active} onClick={() => setSelected(c.label)}>
                    <span className="item__rank num">{i + 1}</span>
                    <span className="item__main">
                      <span className="item__title">
                        {cp.ansatz}
                        {rec?.label === c.label && <span className="pill pill--ok">recommended</span>}
                      </span>
                      <span className="item__sub small muted">{cp.optimizer} · {cp.mitigation}</span>
                      <MiniBar c={c} max={maxMean} />
                    </span>
                    <span className="item__val num">{mha(c.mean)}<span className="muted small"> mHa</span></span>
                  </button>
                </li>
              );
            })}
          </ul>
          {study.configs.length > SHOW && (
            <button type="button" className="btn btn--ghost btn--small" style={{ marginTop: 12 }} onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show fewer" : `Show all ${study.configs.length}`}
            </button>
          )}
        </Panel>

        <Panel
          title={shortName(current)}
          aside={isRec ? <span className="pill pill--ok">recommended</span> : `${p.mapping} · ${p.budget}`}
        >
          <SeedStrip config={current} chemAcc={CHEM_ACC_MHA} hfError={study.hf_error_mha} />
          <dl className="kv" style={{ marginTop: 14 }}>
            <div><dt>Mean error</dt><dd className="num">{mha(current.mean)} ± {mha(current.std)} mHa</dd></div>
            <div><dt>Median / worst</dt><dd className="num">{mha(current.median)} / {mha(current.worst)} mHa</dd></div>
            <div>
              <dt>Within {CHEM_ACC_MHA} mHa</dt>
              <dd className="num">{current.n_chem} of {current.n_seeds} seeds</dd>
              <dd className="small faint num">95% interval {pct(current.p_chem_lo)} to {pct(current.p_chem_hi)}</dd>
            </div>
            <div>
              <dt>Cost</dt>
              <dd className="num">{Math.round(current.two_qubit_gates)} two-qubit gates</dd>
              <dd className="small faint num">{(current.total_shots / 1e6).toFixed(1)}M shots in total</dd>
            </div>
          </dl>
        </Panel>
      </div>

      {study.molecule === "LiH" && <LihPanel studies={studies} />}
    </div>
  );
}
