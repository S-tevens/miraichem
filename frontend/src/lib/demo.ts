/**
 * Offline demo mode: the same interface, with every answer read from a static snapshot of the API
 * (see src/miraichem/api/demo.py, which writes it and defines the file names this mirrors).
 */
export const DEMO = import.meta.env.VITE_DEMO === "1";

// Query parameters that change an answer, per endpoint; the rest is dropped (the demo always shows
// the published results at the default weights).
const PARAMS: Record<string, string[]> = {
  "/api/rank": ["molecule", "backend", "bond_length"],
  "/api/scan": ["molecule", "backend"],
  "/api/replicated/recommendation": ["molecule", "backend", "device"],
  "/api/trust": ["kind", "molecule", "backend", "device"],
};

function number(v: string): string {
  const n = Number(v);
  return Number.isFinite(n) ? String(n) : v;
}

/** File name (relative to demo/) holding the answer to a GET of `url`. */
export function demoFile(url: string): string {
  const [path, query = ""] = url.split("?");
  if (path.startsWith("/api/runs/")) return `runs/${path.slice("/api/runs/".length)}.json`;
  const keep = PARAMS[path] ?? [];
  const params = new URLSearchParams(query);
  const items = [...params.entries()]
    .filter(([k, v]) => keep.includes(k) && v !== "")
    .map(([k, v]) => [k, k === "bond_length" ? number(v) : v] as const)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  const key = path + (items.length ? `?${items.map(([k, v]) => `${k}=${v}`).join("&")}` : "");
  return `${key.replace(/[^A-Za-z0-9._=-]+/g, "_")}.json`;
}

export const DEMO_READ_ONLY =
  "This is the offline demo, which is read-only. Starting runs, sweeps, training or forecasts needs the live app (miraichem serve).";
