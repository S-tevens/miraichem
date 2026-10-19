import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, useQuery } from "../api";
import type { Job, RunRequest, Source, SourceId } from "../types";

// ---------------------------------------------------------------- data source

interface SourceState {
  source: SourceId;
  sources: Source[];
  setSource: (s: SourceId) => void;
  connected: boolean | null; // null = still checking
  connectError: string | null;
  refresh: () => void;
}

const SourceCtx = createContext<SourceState | null>(null);
export const useSource = () => useContext(SourceCtx)!;

const KEY = "miraichem.source";

// ---------------------------------------------------------------- jobs

interface JobsState {
  jobs: Job[];
  active: number;
  startRun: (req: RunRequest) => Promise<Job>;
  startSweep: (name: string, workers: number, force: boolean) => Promise<Job>;
  job: (id: string | null) => Job | undefined;
  /** Bumps whenever a job finishes, so pages can refetch their results. */
  finishedCount: number;
}

const JobsCtx = createContext<JobsState | null>(null);
export const useJobs = () => useContext(JobsCtx)!;

// ---------------------------------------------------------------- run drawer

interface DrawerState {
  hash: string | null;
  source: SourceId | null;
  open: (hash: string, source?: SourceId) => void;
  close: () => void;
}

const DrawerCtx = createContext<DrawerState | null>(null);
export const useDrawer = () => useContext(DrawerCtx)!;

export function AppProviders({ children }: { children: ReactNode }) {
  const sourcesQ = useQuery<{ sources: Source[]; default: SourceId }>("/api/sources");
  const [chosen, setChosen] = useState<SourceId | null>(() => {
    try {
      const v = localStorage.getItem(KEY);
      return v === "local" || v === "published" ? v : null;
    } catch {
      return null;
    }
  });
  const sources = sourcesQ.data?.sources ?? [];
  const source: SourceId = chosen ?? sourcesQ.data?.default ?? "published";

  const setSource = useCallback((s: SourceId) => {
    setChosen(s);
    try {
      localStorage.setItem(KEY, s);
    } catch {
      /* storage can be blocked; the choice just won't persist */
    }
  }, []);

  // ---- jobs: poll quickly while something runs, slowly otherwise
  const [jobs, setJobs] = useState<Job[]>([]);
  const [finishedCount, setFinished] = useState(0);
  const active = jobs.filter((j) => j.status === "queued" || j.status === "running").length;

  const poll = useCallback(async () => {
    try {
      const r = await api.get<{ jobs: Job[] }>("/api/jobs");
      setJobs((prev) => {
        const wasActive = new Set(
          prev.filter((j) => j.status === "queued" || j.status === "running").map((j) => j.id),
        );
        const nowDone = r.jobs.filter(
          (j) => wasActive.has(j.id) && (j.status === "done" || j.status === "failed"),
        );
        if (nowDone.length) setFinished((n) => n + nowDone.length);
        return r.jobs;
      });
    } catch {
      /* the connection indicator reports server problems */
    }
  }, []);

  useEffect(() => {
    if (!sourcesQ.data) return;
    poll();
    const id = setInterval(poll, active > 0 ? 1200 : 8000);
    return () => clearInterval(id);
  }, [poll, active, sourcesQ.data]);

  const startRun = useCallback(
    async (req: RunRequest) => {
      const j = await api.startRun(req);
      await poll();
      return j;
    },
    [poll],
  );
  const startSweep = useCallback(
    async (name: string, workers: number, force: boolean) => {
      const j = await api.startSweep(name, workers, force);
      await poll();
      return j;
    },
    [poll],
  );

  // A finished job changes the run counts shown in the sidebar, so refresh them.
  useEffect(() => {
    if (finishedCount > 0) sourcesQ.reload();
  }, [finishedCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const [drawer, setDrawer] = useState<{ hash: string; source: SourceId | null } | null>(null);

  const srcValue = useMemo<SourceState>(
    () => ({
      source,
      sources,
      setSource,
      connected: sourcesQ.error ? false : sourcesQ.data ? true : null,
      connectError: sourcesQ.error,
      refresh: sourcesQ.reload,
    }),
    [source, sources, setSource, sourcesQ.error, sourcesQ.data, sourcesQ.reload],
  );
  const jobsValue = useMemo<JobsState>(
    () => ({
      jobs,
      active,
      startRun,
      startSweep,
      job: (id) => jobs.find((j) => j.id === id),
      finishedCount,
    }),
    [jobs, active, startRun, startSweep, finishedCount],
  );
  const drawerValue = useMemo<DrawerState>(
    () => ({
      hash: drawer?.hash ?? null,
      source: drawer?.source ?? null,
      open: (hash, s) => setDrawer({ hash, source: s ?? null }),
      close: () => setDrawer(null),
    }),
    [drawer],
  );

  return (
    <SourceCtx.Provider value={srcValue}>
      <JobsCtx.Provider value={jobsValue}>
        <DrawerCtx.Provider value={drawerValue}>{children}</DrawerCtx.Provider>
      </JobsCtx.Provider>
    </SourceCtx.Provider>
  );
}
