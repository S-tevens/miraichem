import { deviceName, mha, moleculeName, parts, when } from "../lib/format";
import { CHEM_ACC_MHA, type RunRow } from "../types";

/** A plain table of runs; clicking a row opens its details. */
export function RunsTable({
  rows,
  onOpen,
  compact = false,
}: {
  rows: RunRow[];
  onOpen: (hash: string) => void;
  compact?: boolean;
}) {
  if (!rows.length) return <p className="small muted">No runs to show.</p>;
  return (
    <div className="tablewrap">
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Molecule</th>
            <th scope="col">Configuration</th>
            <th scope="col">Backend</th>
            <th scope="col" className="right">
              Error (mHa)
            </th>
            {!compact && (
              <>
                <th scope="col" className="right">
                  2q gates
                </th>
                <th scope="col" className="right">
                  Evaluations
                </th>
              </>
            )}
            <th scope="col">Saved</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const p = parts(r);
            return (
              <tr
                key={r.hash}
                data-click
                tabIndex={0}
                onClick={() => onOpen(r.hash)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onOpen(r.hash)}
              >
                <td className="num">
                  {moleculeName(r.molecule)} · {r.bond_length} Å
                </td>
                <td>
                  {p.ansatz}
                  <span className="muted small"> · {p.optimizer} · {p.mitigation}</span>
                </td>
                <td>{r.backend === "ideal" ? "Ideal" : deviceName(r.device)}</td>
                <td className="right num">
                  {r.status === "failed" ? (
                    <span className="pill pill--bad">failed</span>
                  ) : (
                    <>
                      {mha(r.error_mha)}
                      {r.error_mha !== null && r.error_mha < CHEM_ACC_MHA && (
                        <span className="pill pill--ok" style={{ marginLeft: 8 }}>
                          ✓
                        </span>
                      )}
                    </>
                  )}
                </td>
                {!compact && (
                  <>
                    <td className="right num">{r.gates ?? "n/a"}</td>
                    <td className="right num">{r.evals ?? "n/a"}</td>
                  </>
                )}
                <td className="small muted">{when(r.timestamp)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
