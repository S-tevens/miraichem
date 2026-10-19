import { useCallback, useEffect, useRef, useState } from "react";
import { DEMO, DEMO_READ_ONLY, demoFile } from "./lib/demo";
import type { Job, RunRequest } from "./types";

/** Error from the API with the server's own explanation, so the UI can show it as written. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (DEMO) {
    if (init?.method && init.method !== "GET") throw new ApiError(403, DEMO_READ_ONLY);
    if (path.startsWith("/api/jobs")) return { jobs: [] } as T; // nothing can run in the demo
    const res = await fetch(`${import.meta.env.BASE_URL}demo/${demoFile(path)}`);
    if (!res.ok) throw new ApiError(404, "This view is not part of the offline demo.");
    return res.json() as Promise<T>;
  }
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Cannot reach the MiraiChem server. Start it with `miraichem serve`.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  startRun: (req: RunRequest) =>
    request<Job>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  startSweep: (name: string, max_workers: number, force: boolean) =>
    request<Job>(`/api/sweeps/${encodeURIComponent(name)}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_workers, force }),
    }),
};

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && v !== "") u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : "";
}

export interface Query<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Fetches `path` (null = do nothing) and refetches when it changes. Stale answers are dropped. */
export function useQuery<T>(path: string | null): Query<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);
  const latest = useRef(0);

  useEffect(() => {
    if (path === null) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const id = ++latest.current;
    setLoading(true);
    setError(null);
    api
      .get<T>(path)
      .then((d) => id === latest.current && setData(d))
      .catch((e: Error) => {
        if (id === latest.current) {
          setError(e.message);
          setData(null);
        }
      })
      .finally(() => id === latest.current && setLoading(false));
  }, [path, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}
