import { useMemo } from "react";
import { log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { Panel } from "../ui/ui";
import { CHEM_ACC_MHA, type SqdControl, type SqdGroup, type SqdRow } from "../types";
import "./charts.css";
import "./SqdScaling.css";
import "./SqdControls.css";

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const molName = (n: string) => {
  const [head, ...rest] = n.split("_");
  const f = head.replace(/\d/g, (d) => SUB[Number(d)]);
  return rest.length ? `${f} (${rest.join(" ")})` : f;
};

interface Row {
  key: string;
  label: string;
  sub: string;
  value: number;
  std?: number;
  tone: string;
}

const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN);
const sd = (v: number[]) => {
  if (v.length < 2) return 0;
  const m = mean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
};

/** One dot per way of getting a subspace, on a log error axis. */
function Compare({ rows }: { rows: Row[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const compact = width < 560;
  const domain: [number, number] = [0.01, 1000];
  const left = compact ? 14 : 270;
  const right = compact ? 14 : 120;
  const rowH = compact ? 62 : 38;
  const top = 22;
  const x = log(domain, [left, Math.max(left + 140, width - right)]);
  const bottom = top - 6 + rows.length * rowH;
  const height = bottom + 40;
  const ticks = logTicks(domain).filter((t) => Math.log10(t) % 1 === 0);
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={height} role="img" aria-label="Error of SQD and of the same loop with the circuit replaced">
        <rect x={x(domain[0])} y={top - 6} width={x(CHEM_ACC_MHA) - x(domain[0])} height={rows.length * rowH} className="chart__band" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 6} y2={bottom} className="chart__grid" />
            <text x={x(t)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{t >= 1000 ? `${t / 1000}k` : t}</text>
          </g>
        ))}
        {rows.map((r, k) => {
          const cy = top + k * rowH + (compact ? rowH - 18 : rowH / 2 - 6);
          const lo = Math.max(domain[0], r.value - (r.std ?? 0));
          const hi = r.value + (r.std ?? 0);
          return (
            <g key={r.key} className={`scl scl--${r.tone}`}>
              <text x={compact ? left : left - 12} y={compact ? cy - 30 : cy + 4} textAnchor={compact ? "start" : "end"} className="ctl__row">{r.label}</text>
              {compact && <text x={left} y={cy - 16} className="ctl__sub">{r.sub}</text>}
              {!compact && <text x={left - 12} y={cy + 17} textAnchor="end" className="ctl__sub">{r.sub}</text>}
              {r.std !== undefined && r.std > 0 && <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} className="scl__bar" />}
              <circle cx={x(Math.max(r.value, domain[0]))} cy={cy} r={6} className="scl__dot" />
              <text x={Math.min(x(Math.max(r.value, domain[0])) + 12, width - 4)} y={cy - 10} className="scl__label num">{mha(r.value)}</text>
            </g>
          );
        })}
        <text x={x(domain[1])} y={height - 4} textAnchor="end" className="chart__axis">error against the exact energy, mHa (log scale)</text>
      </svg>
    </div>
  );
}

interface Line {
  key: string;
  label: string;
  tone: string;
  pts: { x: number; y: number }[];
}

/** Error against the number of strings kept per spin, with the circuit's own runs as dots. */
function BySize({ control, lines, dots }: { control: SqdControl; lines: Line[]; dots: { x: number; y: number; tone: string }[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const left = 54;
  const right = 150;
  const top = 14;
  const height = 280;
  const bottom = height - 42;
  const xMax = Math.max(...lines.flatMap((l) => l.pts.map((p) => p.x)), ...dots.map((d) => d.x));
  const domainY: [number, number] = [0.001, 1000];
  const x = log([1, xMax * 1.15], [left, Math.max(left + 160, width - right)]);
  const y = log(domainY, [bottom, top]);
  const yTicks = logTicks(domainY).filter((t) => Math.log10(t) % 1 === 0);
  const xTicks = [1, 3, 10, 30, 100, 300].filter((t) => t <= xMax * 1.15);
  const xEnd = x(xMax * 1.15);
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={height} role="img" aria-label="Error against subspace size for random, low-excitation, oracle and circuit-chosen subspaces">
        <rect x={left} y={y(CHEM_ACC_MHA)} width={Math.max(0, xEnd - left)} height={Math.max(0, y(domainY[0]) - y(CHEM_ACC_MHA))} className="chart__band" />
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={left} x2={xEnd} y1={y(t)} y2={y(t)} className="chart__grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">{t >= 1000 ? `${t / 1000}k` : t}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{t}</text>
        ))}
        <line x1={left} x2={xEnd} y1={y(control.hf_error_mha)} y2={y(control.hf_error_mha)} className="chart__hf" />
        <line x1={left} x2={xEnd} y1={y(control.ccsd_error_mha)} y2={y(control.ccsd_error_mha)} className="chart__exact" strokeDasharray="2 4" />
        <text x={xEnd + 8} y={y(control.hf_error_mha) + 4} className="ctl__ref">Hartree-Fock</text>
        <text x={xEnd + 8} y={y(control.ccsd_error_mha) + 4} className="ctl__ref">CCSD</text>
        <text x={x(Math.sqrt(xMax))} y={height - 6} textAnchor="middle" className="chart__axis">strings kept per spin (the subspace has this many squared)</text>
        {lines.map((l) => {
          const pts = l.pts.filter((p) => p.y > 0).sort((a, b) => a.x - b.x);
          if (pts.length < 2) return null;
          const last = pts[pts.length - 1];
          return (
            <g key={l.key} className={`scl scl--${l.tone}`}>
              <polyline points={pts.map((p) => `${x(p.x)},${y(Math.max(p.y, domainY[0]))}`).join(" ")} className="scl__line" />
              <text x={xEnd + 8} y={y(Math.max(last.y, domainY[0])) + 4} className="scl__label">{l.label}</text>
            </g>
          );
        })}
        {dots.map((d, i) => (
          <circle key={i} cx={x(d.x)} cy={y(Math.max(d.y, domainY[0]))} r={4.5} className={`ctl__dot ctl__dot--${d.tone}`} />
        ))}
        <text x={left} y={top - 2} className="chart__axis">error against the exact energy, mHa (log scale)</text>
      </svg>
    </div>
  );
}

const pctSpace = (g: SqdGroup) => {
  const f = 100 * g.mean_subspace_fraction;
  return `${f < 1 ? f.toFixed(2) : f.toFixed(0)}%`;
};

const sameStudy = (g: SqdGroup) =>
  g.noise_model === null && g.readout_noise === 0.02 && g.iterations === 3 && g.samples_per_batch === 300 && g.num_batches === 3 && g.shots === 20000;

/**
 * What does the circuit add? The same SQD loop is run with the circuit replaced (a noisy Hartree-Fock
 * state, or random bits), and the circuit's subspace is compared with subspaces of the same size
 * picked without a circuit. Every error is against the exact energy.
 */
/** Real-device runs next to the ideal simulation of the very same circuit, which the run saved. */
function HardwareVsIdeal({ runs }: { runs: SqdRow[] }) {
  const n = (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : v.toLocaleString("en"));
  const withIdeal = runs.filter((r) => r.hardware?.ideal_same_circuit);
  return (
    <>
      <h3 className="scl__h" style={{ marginTop: 18 }}>Real device against an ideal simulation of the same circuit</h3>
      <div className="tablewrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Device run</th>
              <th scope="col" className="right">Distinct bitstrings</th>
              <th scope="col" className="right">Space solved</th>
              <th scope="col" className="right">Error on the device</th>
              <th scope="col" className="right">Error, ideal circuit</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const ideal = r.hardware?.ideal_same_circuit;
              return (
                <tr key={r.hash}>
                  <td>
                    {r.hardware?.backend}
                    <div className="small faint num">{r.hardware?.job_id}</div>
                  </td>
                  <td className="right num">
                    {n(r.distinct_samples)}
                    {ideal && <span className="small faint"> vs {n(ideal.distinct_samples)}</span>}
                  </td>
                  <td className="right num">{r.subspace_fraction !== null ? `${(100 * r.subspace_fraction).toFixed(r.subspace_fraction < 0.01 ? 4 : 0)}%` : "n/a"}</td>
                  <td className="right num">{r.error_mha !== null ? `${mha(r.error_mha)} mHa` : `${mha(r.below_hf_mha)} mHa below HF`}</td>
                  <td className="right num">{ideal ? (ideal.error_mha !== null ? `${mha(ideal.error_mha)} mHa` : "n/a") : "not simulated"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {withIdeal.length > 0 && (
        <p className="small muted" style={{ marginTop: 10, maxWidth: "78ch" }}>
          An ideal simulation of this circuit gives only {Math.min(...withIdeal.map((r) => r.hardware!.ideal_same_circuit!.distinct_samples))} to{" "}
          {Math.max(...withIdeal.map((r) => r.hardware!.ideal_same_circuit!.distinct_samples))} distinct bitstrings, too few to span a useful subspace. The real
          device&apos;s noise spreads the samples over thousands, which is what lets the subspace solve reach a low error. So in these runs the noise,
          not the circuit&apos;s ideal output, is doing most of the work. That agrees with the control experiments above, and it is why a hardware
          result at this size is a test of the hardware path, not of the method.
        </p>
      )}
    </>
  );
}

export function SqdControls({
  groups,
  controls,
  molecule,
  defaultBonds,
  hardwareRows,
}: {
  groups: SqdGroup[];
  controls: SqdControl[];
  molecule: string;
  defaultBonds: Record<string, number>;
  hardwareRows: SqdRow[];
}) {
  const data = useMemo(() => {
    const control = controls.find((c) => c.molecule === molecule);
    if (!control) return null;
    const bond = defaultBonds[molecule];
    const study = groups.filter((g) => g.molecule === molecule && sameStudy(g) && (bond === undefined || Math.abs(g.bond_length - bond) < 1e-6));
    const circuit = study.find((g) => g.sampler === "circuit" && g.n_reps === 1);
    if (!circuit) return null;
    const hfNoise = study.find((g) => g.sampler === "hf_noise");
    const uniform = study.find((g) => g.sampler === "uniform");
    const one = control.sqd.filter((s) => s.n_reps === 1);
    const eq = one.length
      ? {
          k: mean(one.map((s) => s.k)),
          sqd: mean(one.map((s) => s.error_mha)),
          sqdStd: sd(one.map((s) => s.error_mha)),
          seeds: one.length,
          random: mean(one.map((s) => s.random_mean_mha)),
          randomStd: sd(one.map((s) => s.random_mean_mha)),
          excitation: mean(one.map((s) => s.excitation_mha)),
          oracle: mean(one.map((s) => s.oracle_mha)),
        }
      : null;
    const rows: Row[] = [
      { key: "hf", label: "Hartree-Fock", sub: "classical starting point", value: control.hf_error_mha, tone: "hf" },
      { key: "cc", label: "CCSD", sub: "classical standard method", value: control.ccsd_error_mha, tone: "cc" },
      ...(uniform ? [{ key: "uni", label: "Random bits in place of the circuit", sub: `whole loop, solves ${pctSpace(uniform)} of the space`, value: uniform.mean_error_mha, std: uniform.std_error_mha, tone: "exact" }] : []),
      ...(hfNoise ? [{ key: "hfn", label: "Noisy Hartree-Fock state in place of the circuit", sub: `whole loop, solves ${pctSpace(hfNoise)} of the space`, value: hfNoise.mean_error_mha, std: hfNoise.std_error_mha, tone: "exact" }] : []),
      { key: "q", label: "SQD with the circuit (1 layer)", sub: `whole loop, solves ${pctSpace(circuit)} of the space`, value: eq ? eq.sqd : circuit.mean_error_mha, std: eq ? eq.sqdStd : circuit.std_error_mha, tone: "q" },
      ...(eq
        ? [
            { key: "rs", label: "Random subspace, same size", sub: `about ${Math.round(eq.k)} strings per spin`, value: eq.random, std: eq.randomStd, tone: "exact" },
            { key: "ex", label: "Low-excitation subspace, same size", sub: "what a classical truncation keeps", value: eq.excitation, tone: "exact" },
            { key: "or", label: "Best possible subspace, same size", sub: "needs the exact answer: a bound, not a method", value: eq.oracle, tone: "cc" },
          ]
        : []),
    ];
    const lines: Line[] = [
      { key: "random", label: "random", tone: "exact", pts: control.curve.map((c) => ({ x: c.k, y: c.random_mean_mha })) },
      { key: "excitation", label: "low excitation", tone: "hf", pts: control.curve.map((c) => ({ x: c.k, y: c.excitation_mha })) },
      { key: "oracle", label: "best possible", tone: "cc", pts: control.curve.map((c) => ({ x: c.k, y: c.oracle_mha })) },
    ];
    const dots = control.sqd.map((s) => ({ x: s.k, y: s.error_mha, tone: s.n_reps === 1 ? "q" : "q2" }));
    // The fair test: subspaces of the same size. (Replacing the circuit inside the whole loop changes
    // how much of the space is solved, so it is reported separately.)
    const sameSize = eq
      ? [["a random subspace", eq.random] as const, ["a low-excitation subspace", eq.excitation] as const]
      : [];
    const better = eq ? sameSize.filter(([, v]) => eq.sqd < v).map(([n]) => n) : [];
    const notBetter = eq ? sameSize.filter(([, v]) => eq.sqd >= v).map(([n]) => n) : [];
    return { control, rows, lines, dots, circuit, eq, better, notBetter, hfNoise, uniform };
  }, [groups, controls, molecule, defaultBonds]);

  if (!data && hardwareRows.length === 0) return null;
  if (!data) {
    return (
      <Panel title="What the real device did" aside={`${molName(molecule)}, ${hardwareRows.length} run(s)`}>
        <HardwareVsIdeal runs={hardwareRows} />
      </Panel>
    );
  }
  const { control, circuit, eq, better, notBetter, hfNoise, uniform } = data;
  return (
    <Panel title="Does the quantum part add value?" aside={`${molName(molecule)}, ${control.qubits} qubits, checked against the exact energy`}>
      <p className="small muted" style={{ marginBottom: 10, maxWidth: "78ch" }}>
        SQD&apos;s quantum step only chooses which determinants to solve in. To see what the circuit contributes, the same loop is run with the
        circuit taken out, and the circuit&apos;s subspace is compared with subspaces of the same size picked without it. Everything is simulated, so a
        result that does not favour the circuit is reported as it is.
      </p>
      <Compare rows={data.rows} />
      <ul className="small muted" style={{ margin: "12px 0", paddingLeft: 18, display: "grid", gap: 4, maxWidth: "78ch" }}>
        {eq && (
          <li>
            <strong>Same size, fair test.</strong> With about {Math.round(eq.k)} strings per spin, the circuit&apos;s subspace is off by {mha(eq.sqd)} mHa
            (first {eq.seeds} seeds); a random subspace of that size by {mha(eq.random)} mHa; a low-excitation subspace by {mha(eq.excitation)} mHa.
            {better.length > 0 && ` The circuit is lower than ${better.join(" and ")}.`}
            {notBetter.length > 0 && ` It is not lower than ${notBetter.join(" or ")}.`}
          </li>
        )}
        {eq && eq.oracle > 0 && (
          <li>
            The best possible subspace of that size would give {mha(eq.oracle)} mHa, so the circuit&apos;s choice is {(eq.sqd / eq.oracle).toFixed(0)}× above that bound
            (the bound uses the exact answer, so no method can reach it in practice).
          </li>
        )}
        {(hfNoise || uniform) && (
          <li>
            <strong>Circuit swapped out of the whole loop.</strong>
            {hfNoise && ` A noisy Hartree-Fock state gives ${mha(hfNoise.mean_error_mha)} mHa, solving ${pctSpace(hfNoise)} of the space.`}
            {uniform && ` Random bits give ${mha(uniform.mean_error_mha)} mHa, solving ${pctSpace(uniform)} of the space.`} These solve a different share of the space
            than the circuit run ({pctSpace(circuit)}), so they are not an equal-size comparison. Later rounds repair bitstrings using the previous round&apos;s
            solution, so the loop can improve even from uninformative samples.
            {uniform && eq && uniform.mean_error_mha <= eq.sqd
              ? ` Random bits do at least as well here${
                  uniform.mean_subspace_fraction > circuit.mean_subspace_fraction ? ", mainly because they end up solving a larger share of the space" : ""
                }; the equal-size comparison above is the fair one.`
              : uniform && eq
                ? " Here the circuit's samples give a lower error than random bits in the same loop."
                : ""}
          </li>
        )}
        <li>
          Only one molecule size, one noise level and one layer count are compared here, and the sampling is simulated, so this does not say whether a real device
          would do better or worse.
        </li>
      </ul>
      <h3 className="scl__h">Error against subspace size</h3>
      <BySize control={control} lines={data.lines} dots={data.dots} />
      <div className="chart__key small">
        <span><i className="sw sw--vqe" /> SQD, 1 layer (each seed)</span>
        <span><i className="sw sw--ok" /> within chemical accuracy ({CHEM_ACC_MHA} mHa)</span>
      </div>
      <p className="small faint" style={{ marginTop: 8, maxWidth: "78ch" }}>
        Larger subspaces are always better, so a method is judged by how low its error is for the number of strings it keeps. The dots show the
        circuit runs, each at the size it happened to find; the lines show subspaces of the same size chosen without a circuit
        ({control.n_random} random draws averaged per size).
      </p>
      {hardwareRows.length > 0 && <HardwareVsIdeal runs={hardwareRows} />}
    </Panel>
  );
}
