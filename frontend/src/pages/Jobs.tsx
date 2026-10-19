import type { Route } from "../app/router";
import { useJobs } from "../app/state";
import { JobCard } from "../components/JobCard";
import { Empty, PageHead, Panel } from "../ui/ui";

export function JobsPage({ go }: { go: (r: Route) => void }) {
  const { jobs } = useJobs();
  return (
    <div className="page">
      <PageHead
        title="Jobs"
        sub="Runs and sweeps started from this app, newest first. Jobs live in the running server: restarting it clears this list, but every finished run stays saved."
      />
      {jobs.length === 0 ? (
        <Panel>
          <Empty
            title="Nothing has been started yet"
            art="spiral"
            action={
              <button type="button" className="btn" onClick={() => go("new-run")}>
                Start a run
              </button>
            }
          >
            Jobs from New run and Sweeps appear here with live progress.
          </Empty>
        </Panel>
      ) : (
        <div className="grid">
          {jobs.map((j) => (
            <Panel key={j.id}>
              <JobCard job={j} onOpenResults={() => go("leaderboard")} />
            </Panel>
          ))}
        </div>
      )}
    </div>
  );
}
