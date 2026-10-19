import { useState } from "react";
import { log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { CHEM_ACC_MHA, type Config, type Study } from "../types";
import { Panel } from "../ui/ui";
import "./LihPanel.css";

interface Row {
  id: string;
  title: string;
  gates: number | null;
  ideal?: Config;
  noisy?: Config;
}

const best = (cs: Config[]): Config | undefined => [...cs].sort((a, b) => a.mean - b.mean)[0];

/** Best configuration per circuit family, for the ideal and the noisy backend. */
function buildRows(studies: Study[]): Row[] {
  const ideal = studies.find((s) => s.molecule === "LiH" && s.backend === "ideal");
  const noisy = studies.find((s) => s.molecule === "LiH" && s.backend === "noisy");
  const families: { id: string; title: string; match: (c: Config) => boolean; layers?: number }[] = [
    { id: "uccsd", title: "UCCSD", match: (c) => c.ansatz === "uccsd" },
    ...[1, 2, 3, 4].map((r) => ({
      id: `hea${r}`,
      title: `Hardware-efficient, ${r} ${r === 1 ? "layer" : "layers"}`,
      match: (c: Config) => c.ansatz !== "uccsd" && c.reps === r,
      layers: r,
    })),
  ];
  return families.map((f) => {
    const i = best(ideal?.configs.filter(f.match) ?? []);
    const n = best(noisy?.configs.filter(f.match) ?? []);
    // Two-qubit gates come from the transpiled noisy run; for ideal-only circuits the
    // hardware-efficient pattern on 4 qubits has 3 entangling gates per layer.
    const gates = n ? Math.round(n.two_qubit_gates) : f.layers ? 3 * f.layers : null;
    return { id: f.id, title: f.title, gates, ideal: i, noisy: n };
  });
}

export function LihPanel({ studies }: { studies: Study[] }) {
  const rows = buildRows(studies).filter((r) => r.ideal || r.noisy);
  const hf = studies.find((s) => s.molecule === "LiH")?.hf_error_mha ?? undefined;
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  if (!rows.length || hf === undefined) return null;

  const ca = CHEM_ACC_MHA;
  const bestNoisy = best(rows.flatMap((r) => (r.noisy ? [r.noisy] : [])));
  const deep = rows.find((r) => r.id === "hea4")?.ideal;
  const idealStudy = studies.find((st) => st.molecule === "LiH" && st.backend === "ideal");
  const spsaDeep = best(idealStudy?.configs.filter((c) => c.optimizer.startsWith("spsa") && c.reps >= 3) ?? []);

  const domain: [number, number] = [0.1, 1000];
  const compact = width < 600; // labels sit above each track on narrow screens (HEA = hardware-efficient)
  const left = compact ? 14 : 210;
  const right = 20;
  const rowH = compact ? 72 : 58;
  const top = 44;
  const height = top + rows.length * rowH + 44;
  const x = log(domain, [left, Math.max(left + 150, width - right)]);
  const ticks = logTicks(domain).filter((t) => [0.1, 1, 10, 100, 1000].includes(t));

  const tip = (r: Row, which: "ideal" | "noisy") => {
    const c = r[which];
    return c ? `${r.title}, ${which === "ideal" ? "ideal simulator" : "noisy simulator"}: ${mha(c.mean)} mHa (mean of ${c.n_seeds} seeds)` : "";
  };

  return (
    <Panel title="Lithium hydride: circuit depth against noise" aside="best configuration per circuit, mean over seeds">
      <div className="lih">
        <p className="muted lih__lede">
          Hartree-Fock alone is off by {mha(hf)} mHa for this molecule, so a quantum result only means something if
          it beats that.
          {deep && <> With no noise, a hardware-efficient circuit needs four layers to get there ({mha(deep.mean)} mHa).</>}
          {bestNoisy && <> On the noisy simulator, the best configuration averages {mha(bestNoisy.mean)} mHa.</>}
        </p>

        <div ref={ref} className="lih__chart">
          <svg width={width} height={height} role="img" aria-label="Best mean error per circuit, ideal versus noisy simulator, for lithium hydride">
            <rect x={x(domain[0])} y={top - 10} width={x(ca) - x(domain[0])} height={rows.length * rowH} className="lih__band" />
            <line x1={x(hf)} x2={x(hf)} y1={top - 10} y2={top - 10 + rows.length * rowH} className="lih__hf" />
            <text x={x(hf) + 6} y={top - 18} className="lih__ref lih__ref--hf">
              Hartree-Fock, {mha(hf)}
            </text>
            <text x={x(domain[0])} y={top - 18} className="lih__ref lih__ref--ok">
              chemical accuracy
            </text>
            {rows.map((r, k) => {
              const cy = top + k * rowH + (compact ? rowH - 22 : rowH / 2 - 10);
              return (
                <g key={r.id}>
                  <text
                    x={compact ? left : left - 16}
                    y={compact ? cy - 22 : cy - 2}
                    textAnchor={compact ? "start" : "end"}
                    className="lih__label"
                  >
                    {compact ? r.title.replace("Hardware-efficient", "HEA") : r.title}
                    {compact && r.gates !== null ? ` · ${r.gates} two-qubit gates` : ""}
                  </text>
                  {!compact && r.gates !== null && (
                    <text x={left - 16} y={cy + 14} textAnchor="end" className="lih__gates num">
                      {r.gates} two-qubit gates
                    </text>
                  )}
                  <line x1={x(domain[0])} x2={x(domain[1])} y1={cy} y2={cy} className="lih__track" />
                  {r.ideal && r.noisy && (
                    <line x1={x(r.ideal.mean)} x2={x(r.noisy.mean)} y1={cy} y2={cy} className="lih__link" />
                  )}
                  {r.ideal && (
                    <circle
                      cx={x(r.ideal.mean)}
                      cy={cy}
                      r={hover === `${r.id}-i` ? 9 : 7.5}
                      className="lih__dot lih__dot--ideal"
                      tabIndex={0}
                      aria-label={tip(r, "ideal")}
                      onMouseEnter={() => setHover(`${r.id}-i`)}
                      onMouseLeave={() => setHover(null)}
                      onFocus={() => setHover(`${r.id}-i`)}
                      onBlur={() => setHover(null)}
                    />
                  )}
                  {r.noisy && (
                    <circle
                      cx={x(r.noisy.mean)}
                      cy={cy}
                      r={hover === `${r.id}-n` ? 9 : 7.5}
                      className="lih__dot lih__dot--noisy"
                      tabIndex={0}
                      aria-label={tip(r, "noisy")}
                      onMouseEnter={() => setHover(`${r.id}-n`)}
                      onMouseLeave={() => setHover(null)}
                      onFocus={() => setHover(`${r.id}-n`)}
                      onBlur={() => setHover(null)}
                    />
                  )}
                </g>
              );
            })}
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top - 10 + rows.length * rowH} y2={top - 4 + rows.length * rowH} className="lih__tick" />
                <text x={x(t)} y={top + 16 + rows.length * rowH} textAnchor="middle" className="lih__axis num">
                  {t}
                </text>
              </g>
            ))}
            <text x={width - right} y={height - 2} textAnchor="end" className="lih__axis">
              mean error in mHa, log scale
            </text>
          </svg>
          <div className="lih__key small">
            <span>
              <i className="lih__swatch lih__swatch--ideal" /> Ideal simulator (exact gradients where noted)
            </span>
            <span>
              <i className="lih__swatch lih__swatch--noisy" /> Noisy simulator (shot noise: SPSA only)
            </span>
          </div>
          <p className="lih__live small" aria-live="polite">
            {hover
              ? (() => {
                  const r = rows.find((q) => hover.startsWith(q.id));
                  return r ? tip(r, hover.endsWith("-i") ? "ideal" : "noisy") : "";
                })()
              : "Hover or focus a dot to read its value."}
          </p>
        </div>

        <p className="small muted lih__note">
          Each dot is the best configuration for that circuit, averaged over its seeds. The ideal
          four-layer result uses exact gradients (L-BFGS-B), which are not available when energies
          come from a finite number of shots; with SPSA, the only optimizer that works on noisy
          devices, the best ideal result for the deeper circuits averages{" "}
          {spsaDeep ? `${mha(spsaDeep.mean)} mHa` : "more than chemical accuracy"}. UCCSD is exact on the
          ideal backend too, but that run was not repeated over seeds, so it is not plotted. We
          report this as a limit of today&apos;s noisy devices, not a problem we have solved.
        </p>
      </div>
    </Panel>
  );
}
