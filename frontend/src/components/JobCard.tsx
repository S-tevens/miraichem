import { duration, mha } from "../lib/format";
import { useDrawer } from "../app/state";
import { Progress } from "../ui/ui";
import type { Job } from "../types";

/** One job: status, progress, time estimate and a way to its results. */
export function JobCard({ job, onOpenResults }: { job: Job; onOpenResults?: () => void }) {
  const drawer = useDrawer();
  const active = job.status === "queued" || job.status === "running";
  const done = job.completed - job.cached;
  const remaining = Math.max(0, job.total - job.completed);
  const rate = done > 0 && job.elapsed_s > 0 ? job.elapsed_s / done : null;
  const eta = active && rate && remaining > 0 ? rate * remaining : null;

  return (
    <article className="jobcard" style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
        <strong style={{ fontWeight: 400 }}>{job.title}</strong>
        <span className={`pill ${job.status === "done" ? "pill--ok" : job.status === "failed" ? "pill--bad" : ""}`}>
          {job.status}
        </span>
      </div>
      {job.kind === "sweep" && (
        <>
          <Progress value={job.completed} total={job.total} tone={job.status === "failed" ? "bad" : job.status === "done" ? "ok" : undefined} />
          <p className="small muted num">
            {job.completed} of {job.total} runs
            {job.cached ? ` (${job.cached} were already saved)` : ""}
            {job.failed_runs ? ` · ${job.failed_runs} failed` : ""}
            {` · ${duration(job.elapsed_s)} elapsed`}
            {eta ? ` · about ${duration(eta)} left` : ""}
            {job.best_error_mha !== null ? ` · best error so far ${mha(job.best_error_mha)} mHa` : ""}
          </p>
        </>
      )}
      {job.kind === "run" && (
        <p className="small muted num">
          {duration(job.elapsed_s)}
          {job.best_error_mha !== null ? ` · error ${mha(job.best_error_mha)} mHa` : ""}
        </p>
      )}
      {job.message && !job.error && <p className="small faint">{job.message}</p>}
      {job.error && <p className="small" style={{ color: "var(--bad)" }}>{job.error}</p>}
      {!active && job.status === "done" && (
        <div>
          {job.kind === "run" && job.result_hashes[0] ? (
            <button type="button" className="btn btn--ghost btn--small" onClick={() => drawer.open(job.result_hashes[0], "local")}>
              Open run
            </button>
          ) : (
            onOpenResults && (
              <button type="button" className="btn btn--ghost btn--small" onClick={onOpenResults}>
                See the results in the leaderboard
              </button>
            )
          )}
        </div>
      )}
    </article>
  );
}
