import { useState } from "react";
import { qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import { deviceName, mha, moleculeName } from "../lib/format";
import { log } from "../lib/scale";
import { Empty, Loading, Notice, PageHead, Panel, Select } from "../ui/ui";
import { CHEM_ACC_MHA, type DeviceBlock } from "../types";
import "./Devices.css";

const R_MIN = 5;
const R_MAX = 20;

export function DevicesPage() {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const q = useQuery<{ blocks: DeviceBlock[] }>(`/api/devices${qs({ source, n: finishedCount })}`);
  const blocks = q.data?.blocks ?? [];
  const [mol, setMol] = useState("");
  const block = blocks.find((b) => b.molecule === mol) ?? blocks[0];

  if (q.loading && !q.data) return <div className="page"><Loading /></div>;
  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;
  if (!block) {
    return (
      <div className="page">
        <PageHead title="Devices" />
        <Panel>
          <Empty title="No multi-device results in this source" art="stagger">
            Run the <code>h2_devices</code> sweep to compare the same configurations on several simulated IBM devices.
          </Empty>
        </Panel>
      </div>
    );
  }

  const all = block.configs.flatMap((c) => block.devices.map((d) => c.means[d]).filter((v): v is number => v !== null));
  const r = log([Math.min(...all), Math.max(...all)], [R_MIN, R_MAX]);
  const rho = block.devices
    .flatMap((a, i) => block.devices.slice(i + 1).map((b) => block.spearman[a]?.[b]))
    .filter((v): v is number => typeof v === "number");

  return (
    <div className="page">
      <PageHead
        title="Devices"
        sub="Each IBM device has its own error profile. The same configurations were run on simulated versions of several devices; if the order barely changes, a recommendation made on one device is likely to carry over."
        tools={
          blocks.length > 1 ? (
            <Select label="Molecule" value={block.molecule} onChange={setMol} options={blocks.map((b) => ({ id: b.molecule, label: moleculeName(b.molecule) }))} />
          ) : undefined
        }
      />

      <div className="stats">
        <div className="stat">
          <span className="stat__label">Devices compared</span>
          <span className="stat__value num">{block.devices.length}</span>
          <span className="stat__note">{block.devices.map(deviceName).join(", ")}</span>
        </div>
        <div className="stat">
          <span className="stat__label">Rank agreement between devices</span>
          <span className="stat__value num">
            {rho.length ? `${Math.min(...rho).toFixed(2)} to ${Math.max(...rho).toFixed(2)}` : "n/a"}
          </span>
          <span className="stat__note">Spearman correlation; 1 = identical order</span>
        </div>
        <div className="stat">
          <span className="stat__label">Configurations</span>
          <span className="stat__value num">{block.configs.length}</span>
          <span className="stat__note">run on every device</span>
        </div>
      </div>

      <div className="grid devices__body">
        <Panel title="Mean error per configuration and device" aside="mHa, mean over seeds">
          <div className="devices__chart">
            <table className="dgrid" style={{ "--n": block.devices.length } as React.CSSProperties} aria-label="Mean error in mHa per configuration and device">
              <thead>
                <tr>
                  <th scope="col" className="dgrid__corner">Configuration</th>
                  {block.devices.map((d) => (
                    <th scope="col" key={d}>{deviceName(d)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.configs.map((c) => (
                  <tr key={c.name}>
                    <th scope="row" className="dgrid__name">
                      {c.name.replace(/ \[.*\]$/, "").replace(/^parity \/ /, "").replace(/ \/ /g, " · ")}
                    </th>
                    {block.devices.map((d) => {
                      const v = c.means[d];
                      if (v === null) return <td key={d} />;
                      return (
                        <td key={d} title={`${deviceName(d)}: ${mha(v)} mHa`}>
                          <span className={`dgrid__dot ${v < CHEM_ACC_MHA ? "dgrid__dot--ok" : ""}`} style={{ width: 2 * r(v), height: 2 * r(v) }} />
                          <span className="dgrid__val num">{mha(v)}</span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small faint" style={{ marginTop: 12 }}>
            Dot size follows the error on a log scale. Chemical accuracy is {CHEM_ACC_MHA} mHa.
          </p>
        </Panel>

        <Panel title="Recommended on each device">
          <ul className="recs">
            {block.devices.map((d) => {
              const rec = block.recommendations[d];
              return (
                <li key={d}>
                  <span className="recs__dev">{deviceName(d)}</span>
                  <span className="small muted">
                    {rec.label.split(" / ").slice(1, 4).join(" · ").replace(/ \[.*\]$/, "")}
                  </span>
                  <span className="small num">{mha(rec.mean)} ± {mha(rec.std)}</span>
                </li>
              );
            })}
          </ul>
          <p className="small muted" style={{ marginTop: 12 }}>{block.summary}</p>
        </Panel>
      </div>
    </div>
  );
}
