import { useEffect, useState } from "react";

export type Route =
  | "overview"
  | "leaderboard"
  | "reliability"
  | "devices"
  | "curves"
  | "hardware"
  | "predictor"
  | "new-run"
  | "sweeps"
  | "jobs"
  | "about";

export const ROUTES: Route[] = [
  "overview",
  "leaderboard",
  "reliability",
  "devices",
  "curves",
  "hardware",
  "predictor",
  "new-run",
  "sweeps",
  "jobs",
  "about",
];

function read(): Route {
  const h = window.location.hash.replace(/^#\/?/, "") as Route;
  return ROUTES.includes(h) ? h : "overview";
}

/** Tiny hash router: the whole app is one page, so deep links are just `#/leaderboard`. */
export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [
    route,
    (r) => {
      window.location.hash = `#/${r}`;
    },
  ];
}
