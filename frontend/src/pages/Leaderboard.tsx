import { useEffect, useMemo, useState } from "react";
import { qs, useQuery } from "../api";
import { useDrawer, useJobs, useSource } from "../app/state";
import { ParetoChart, type CostAxis } from "../components/ParetoChart";
import { DEMO } from "../lib/demo";
import { deviceName, mha, moleculeName, parts } from "../lib/format";
import { Empty, Field, Loading, Notice, PageHead, Panel, Select, Seg } from "../ui/ui";
import { CHEM_ACC_MHA, type Overview, type RankResponse, type RankedRow } from "../types";
import "./Leaderboard.css";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export function LeaderboardPage() {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const drawer = useDrawer();
  const ov = useQuery<Overview>(`/api/overview${qs({ source, n: finishedCount })}`);

  const molecules = Object.keys(ov.data?.by_molecule ?? {});
  const [molecule, setMolecule] = useState("");
  const [backend, setBackend] = useState<"ideal" | "noisy">("noisy");
  const [bond, setBond] = useState<string>("");
  const [w, setW] = useState({ acc: 70, gates: 20, shots: 10 });
  const wd = useDebounced(w, 250);
  const [axis, setAxis] = useState<CostAxis>("gates");
  const [fAnsatz, setFAnsatz] = useState("");
  const [fOpt, setFOpt] = useState("");
  const [fMit, setFMit] = useState("");

  // pick sensible defaults once the overview is in
  useEffect(() => {
    if (!molecule && molecules.length) setMolecule(molecules.includes("H2") ? "H2" : molecules[0]);
  }, [molecules, molecule]);
  const backends = Object.keys(ov.data?.by_molecule[molecule] ?? {}) as ("ideal" | "noisy")[];
  useEffect(() => {
    if (backends.length && !backends.includes(backend)) setBackend(backends[0]);
  }, [backends, backend]);
  useEffect(() => {
    setBond("");
    setFAnsatz("");
    setFOpt("");
    setFMit("");
  }, [molecule, backend, source]);

  const hasWeight = wd.acc + wd.gates + wd.shots > 0;
  const path = molecule && hasWeight
    ? `/api/rank${qs({ molecule, backend, bond_length: bond, w_acc: wd.acc, w_gates: wd.gates, w_shots: wd.shots, source, n: finishedCount })}`
    : null;
  const rank = useQuery<RankResponse>(path);
  const data = rank.data;

  const rows = useMemo(() => {
    const all = data?.ranked ?? [];
    return all.filter(
      (r) => (!fAnsatz || r.ansatz === fAnsatz) && (!fOpt || r.optimizer === fOpt) && (!fMit || r.mitigation === fMit),
    );
  }, [data, fAnsatz, fOpt, fMit]);
  const distinct = (k: keyof RankedRow) => [...new Set((data?.ranked ?? []).map((r) => String(r[k])))].sort();
  const maxErr = Math.max(...rows.map((r) => r.error_mha ?? 0), 1);
  const noisy = backend === "noisy";

  return (
    <div className="page">
      <PageHead
        title="Leaderboard"
        sub="Every run for one molecule and backend, scored on accuracy, circuit size and shots. Change the weights to see how the ranking and the recommendation move."
        tools={
          <>
            <Select
              label="Molecule"
              value={molecule}
              onChange={setMolecule}
              options={molecules.map((m) => ({ id: m, label: moleculeName(m) }))}
            />
            <Field label="Backend">
              <Seg
                label="Backend"
                value={backend}
                onChange={setBackend}
                options={backends.map((b) => ({ id: b, label: b === "ideal" ? "Ideal" : "Noisy" }))}
              />
            </Field>
            {data && data.bond_lengths.length > 1 && (
              <Select
                label="Bond length (Å)"
                value={bond || String(data.bond_length)}
                onChange={setBond}
                options={data.bond_lengths.map((b) => ({ id: String(b), label: String(b) }))}
              />
            )}
          </>
        }
      />

      {rank.error && (
        <Notice tone="bad">
          {rank.error.includes("No successful runs")
            ? "There are no runs for this selection yet. Try another molecule or backend, or run one."
            : rank.error}
        </Notice>
      )}
      {rank.loading && !data && <Loading />}

      {data && (
        <>
          <div className="grid grid--side">
            <Panel title="Scoring weights" aside={DEMO ? "fixed in the offline demo" : "normalised to 100%"}>
              <div className="weights">
                {(
                  [
                    ["acc", "Accuracy", "how close to the exact energy"],
                    ["gates", noisy ? "Two-qubit gates" : "Circuit depth", "deeper circuits are noisier"],
                    ["shots", "Shots", "quantum time spent"],
                  ] as const
                ).map(([k, name, hint]) => (
                  <label key={k} className="weight">
                    <span className="weight__row">
                      <span>{name}</span>
                      <span className="num muted">{w[k]}</span>
                    </span>
                    <input
                      type="range"
                      className="range"
                      min={0}
                      max={100}
                      step={5}
                      value={w[k]}
                      onChange={(e) => setW((p) => ({ ...p, [k]: Number(e.target.value) }))}
                      aria-label={`${name} weight`}
                      disabled={DEMO}
                    />
                    <span className="field__hint">{hint}</span>
                  </label>
                ))}
                {w.acc + w.gates + w.shots === 0 && <Notice tone="warn">Give at least one weight a value above zero.</Notice>}
              </div>
            </Panel>

            <Panel
              title="Recommendation"
              aside={
                <span className={`pill ${data.reached_chemical_accuracy ? "pill--ok" : "pill--warn"}`}>
                  {data.reached_chemical_accuracy ? "reaches chemical accuracy" : "none reaches chemical accuracy"}
                </span>
              }
            >
              <p className="rec-text">{data.justification}</p>
              <p className="small faint" style={{ marginTop: 10 }}>
                This ranks single runs. For a recommendation that accounts for random-seed luck, see Seed reliability, or open its{" "}
                <a href={`#/trust?kind=vqe&molecule=${encodeURIComponent(molecule)}&backend=${backend}`}>trust report</a>.
              </p>
            </Panel>
          </div>

          <Panel
            title="Error against cost"
            aside={
              backend === "noisy" ? (
                <Seg
                  label="Cost axis"
                  value={axis}
                  onChange={setAxis}
                  options={[
                    { id: "gates", label: "Two-qubit gates" },
                    { id: "shots", label: "Shots" },
                  ]}
                />
              ) : undefined
            }
          >
            <ParetoChart
              rows={rows}
              axis={backend === "noisy" ? axis : "gates"}
              bestHash={data.best_hash}
              onSelect={(h) => drawer.open(h, source)}
            />
          </Panel>

          <Panel title={`All runs (${rows.length})`} aside="click a row for details">
            <div className="filters">
              <Select label="Ansatz" value={fAnsatz} onChange={setFAnsatz} options={[{ id: "", label: "All" }, ...distinct("ansatz").map((v) => ({ id: v, label: v }))]} />
              <Select label="Optimizer" value={fOpt} onChange={setFOpt} options={[{ id: "", label: "All" }, ...distinct("optimizer").map((v) => ({ id: v, label: v }))]} />
              <Select label="Mitigation" value={fMit} onChange={setFMit} options={[{ id: "", label: "All" }, ...distinct("mitigation").map((v) => ({ id: v, label: v }))]} />
            </div>
            {rows.length === 0 ? (
              <Empty title="No runs match these filters" art="grid" />
            ) : (
              <div className="tablewrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">#</th>
                      <th scope="col">Configuration</th>
                      <th scope="col">Error (mHa)</th>
                      <th scope="col" className="right">{noisy ? "2q gates" : "Depth"}</th>
                      <th scope="col" className="right">Shots</th>
                      <th scope="col" className="right">Score</th>
                      <th scope="col">Front</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 300).map((r) => {
                      const p = parts(r);
                      const best = r.hash === data.best_hash;
                      return (
                        <tr
                          key={r.hash}
                          data-click
                          tabIndex={0}
                          className={best ? "is-best" : ""}
                          onClick={() => drawer.open(r.hash, source)}
                          onKeyDown={(e) => e.key === "Enter" && drawer.open(r.hash, source)}
                        >
                          <td className="num muted">{r.rank}</td>
                          <td>
                            <span>{p.ansatz}</span>
                            {best && <span className="pill pill--ok" style={{ marginLeft: 8 }}>recommended</span>}
                            <div className="muted small">
                              {p.mapping} · {p.optimizer} · {p.mitigation}
                              {noisy ? ` · ${deviceName(r.device)}` : ""}
                            </div>
                          </td>
                          <td style={{ minWidth: 150 }}>
                            <span className="num">{mha(r.error_mha)}</span>
                            <div className="bar" style={{ marginTop: 5 }}>
                              <div
                                className={`bar__fill ${(r.error_mha ?? 0) < CHEM_ACC_MHA ? "bar__fill--ok" : ""}`}
                                style={{ width: `${Math.max(2, (100 * Math.log10(1 + (r.error_mha ?? 0))) / Math.log10(1 + maxErr))}%` }}
                              />
                            </div>
                          </td>
                          <td className="right num">{noisy ? r.gates ?? "n/a" : r.depth ?? "n/a"}</td>
                          <td className="right num">{r.total_shots ? r.total_shots.toLocaleString("en") : "exact"}</td>
                          <td className="right num muted">{r.score.toFixed(3)}</td>
                          <td className="small muted">
                            {[r.pareto_gates ? "gates" : "", r.pareto_shots ? "shots" : ""].filter(Boolean).join(", ")}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {rows.length > 300 && <p className="small faint" style={{ marginTop: 8 }}>Showing the best 300 of {rows.length}.</p>}
          </Panel>
        </>
      )}
    </div>
  );
}
