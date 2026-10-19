import { AppProviders, useSource } from "./app/state";
import { useRoute, type Route } from "./app/router";
import { Shell } from "./app/Shell";
import { RunDrawer } from "./components/RunDrawer";
import { Notice } from "./ui/ui";
import { AboutPage } from "./pages/About";
import { CurvesPage } from "./pages/Curves";
import { DevicesPage } from "./pages/Devices";
import { HardwarePage } from "./pages/Hardware";
import { JobsPage } from "./pages/Jobs";
import { LeaderboardPage } from "./pages/Leaderboard";
import { NewRunPage } from "./pages/NewRun";
import { OverviewPage } from "./pages/Overview";
import { PredictorPage } from "./pages/Predictor";
import { ReliabilityPage } from "./pages/Reliability";
import { SweepsPage } from "./pages/Sweeps";

function Page({ route, go }: { route: Route; go: (r: Route) => void }) {
  switch (route) {
    case "overview":
      return <OverviewPage go={go} />;
    case "leaderboard":
      return <LeaderboardPage />;
    case "reliability":
      return <ReliabilityPage />;
    case "devices":
      return <DevicesPage />;
    case "curves":
      return <CurvesPage />;
    case "hardware":
      return <HardwarePage />;
    case "predictor":
      return <PredictorPage />;
    case "new-run":
      return <NewRunPage />;
    case "sweeps":
      return <SweepsPage go={go} />;
    case "jobs":
      return <JobsPage go={go} />;
    case "about":
      return <AboutPage />;
  }
}

function Frame() {
  const [route, go] = useRoute();
  const { connected, connectError } = useSource();
  return (
    <Shell route={route} go={go}>
      {connected === false && (
        <div className="page">
          <Notice tone="bad">
            {connectError ?? "Cannot reach the MiraiChem server."} Start it with <code>miraichem serve</code> and reload this page.
          </Notice>
        </div>
      )}
      {connected !== false && <Page route={route} go={go} />}
      <RunDrawer />
    </Shell>
  );
}

export function App() {
  return (
    <AppProviders>
      <Frame />
    </AppProviders>
  );
}
