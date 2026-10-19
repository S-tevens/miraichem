import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { useDrawer } from "../app/state";
import { deviceName, mha, parts, pct } from "../lib/format";
import { Loading, Notice, Panel } from "../ui/ui";
import type { PredictionResult, RunRequest } from "../types";
import { ForecastChart } from "./ForecastChart";
import "./ParetoChart.css";
import "./PredictionPanel.css";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

/** What the model expects this configuration to do, before spending any compute. */
export function PredictionPanel({ req, actual }: { req: RunRequest; actual?: number | null }) {
  const drawer = useDrawer();
  // everything that changes the forecast; the seed and "force" do not
  const key = JSON.stringify({ ...req, seed: 0, force: false });
  const settled = useDebounced(key, 600);
  const [res, setRes] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let current = true;
    setBusy(true);
    setError(null);
    api
      .post<PredictionResult>("/api/predict", JSON.parse(settled))
      .then((r) => current && setRes(r))
      .catch((e: unknown) => current && (setError(e instanceof ApiError ? e.message : String(e)), setRes(null)))
      .finally(() => current && setBusy(false));
    return () => {
      current = false;
    };
  }, [settled]);

  const pred = res?.prediction;
  return (
    <Panel
      title="Predicted error"
      aside={busy ? <span className="small muted">updating…</span> : res?.origin ? `model: ${res.origin}` : undefined}
    >
      {!res && busy && <Loading label="Preparing the circuit" />}
      {error && !res && (
        <Notice tone={error.includes("No trained predictor") ? undefined : "bad"}>
          {error.includes("No trained predictor")
            ? "No predictor has been trained yet. Open the Error predictor page to train one from your runs."
            : error}
        </Notice>
      )}
      {res && !res.supported && <p className="small muted">{res.reason}</p>}
      {pred && res?.circuit && (
        <div className="forecast">
          <ForecastChart low={pred.low_mha} mid={pred.error_mha} high={pred.high_mha} hf={pred.hf_error_mha} actual={actual} />
          <div className="stats">
            <div className="stat">
              <span className="stat__label">Chance of chemical accuracy</span>
              <span className="stat__value num">{pct(pred.p_chemical_accuracy)}</span>
              <span className="stat__note">for one run with a random seed</span>
            </div>
            <div className="stat">
              <span className="stat__label">Chance of beating Hartree-Fock</span>
              <span className="stat__value num">{pct(pred.p_beats_hartree_fock)}</span>
              <span className="stat__note">HF error is {mha(pred.hf_error_mha)} mHa</span>
            </div>
            <div className="stat">
              <span className="stat__label">Circuit on this device</span>
              <span className="stat__value num">{Math.round(res.circuit.two_qubit_gates)}</span>
              <span className="stat__note">two-qubit gates · depth {Math.round(res.circuit.depth)}</span>
            </div>
          </div>
          {pred.warnings.map((w) => (
            <Notice key={w} tone="warn">{w}</Notice>
          ))}
          {(res.similar?.length ?? 0) > 0 && (
            <div>
              <h3 className="small muted" style={{ marginBottom: 6 }}>Closest runs already measured</h3>
              <div className="tablewrap">
                <table className="table">
                  <tbody>
                    {res.similar!.map((r) => {
                      const p = parts(r);
                      return (
                        <tr key={r.hash} data-click tabIndex={0} onClick={() => drawer.open(r.hash)} onKeyDown={(e) => e.key === "Enter" && drawer.open(r.hash)}>
                          <td>
                            {p.ansatz}
                            <span className="muted small"> · {p.optimizer} · {p.mitigation} · {r.backend === "noisy" ? deviceName(r.device) : ""}</span>
                          </td>
                          <td className="right num">{mha(r.error_mha)} mHa</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <p className="small faint">
            A forecast from earlier runs, not a guarantee. The interval is the range that contained the
            true error 80% of the time when this model was checked on configurations it had not seen.
          </p>
        </div>
      )}
    </Panel>
  );
}
