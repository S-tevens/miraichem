"""Export a read-only snapshot of the API for the offline demo build of the web app.

The demo is the same interface with its data answered from static JSON files, so it needs no
Python, no server and no installation: any file server (or a double-click on a local server such as
``python -m http.server``) can show it. It covers the published results only. Anything that starts
work (runs, sweeps, training, forecasts) needs the live app and is disabled in the demo.

The file name for each answer comes from :func:`demo_key`, which the frontend mirrors in
``frontend/src/lib/demo.ts``. Both are tested against the same literal examples.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from miraichem.api.app import create_app

# Query parameters that change an answer, per endpoint. Everything else (the data source, cache
# busters, scoring weights) is dropped: the demo always shows the published results at the
# default weights.
_PARAMS: dict[str, tuple[str, ...]] = {
    "/api/rank": ("molecule", "backend", "bond_length"),
    "/api/scan": ("molecule", "backend"),
    "/api/replicated/recommendation": ("molecule", "backend", "device"),
}


def _number(value: str) -> str:
    try:
        return f"{float(value):g}"
    except ValueError:
        return value


def demo_key(path: str, query: dict[str, str] | None = None) -> str:
    """The file name (without folder) that holds the answer to ``path`` with ``query``."""
    keep = _PARAMS.get(path, ())
    items = sorted(
        (k, _number(v) if k == "bond_length" else v)
        for k, v in (query or {}).items()
        if k in keep and v not in ("", None)
    )
    key = path + ("?" + "&".join(f"{k}={v}" for k, v in items) if items else "")
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", key) + ".json"


def export_demo(
    out_dir: Path,
    published_dir: Path = Path("results/published"),
    configs_dir: Path = Path("configs"),
) -> dict[str, int]:
    """Write the snapshot into ``out_dir``. Returns how many files of each kind were written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    empty = (
        out_dir.parent / ".demo-empty-local"
    )  # an empty "my runs" folder: demo is published-only
    api = TestClient(create_app(empty, published_dir, configs_dir, None))
    counts = {"views": 0, "rank": 0, "scan": 0, "runs": 0}

    def save(path: str, query: dict[str, Any] | None = None, kind: str = "views") -> Any:
        params = {k: str(v) for k, v in (query or {}).items()}
        res = api.get(path, params={**params, "source": "published"})
        if res.status_code != 200:
            return None
        (out_dir / demo_key(path, params)).write_text(
            json.dumps(res.json(), separators=(",", ":")), encoding="utf-8"
        )
        counts[kind] += 1
        return res.json()

    for path in (
        "/api/sources",
        "/api/options",
        "/api/overview",
        "/api/replicated",
        "/api/devices",
        "/api/hardware",
        "/api/curves",
        "/api/predictor",
        "/api/sweeps",
        "/api/sqd",
        "/api/sqd/options",
    ):
        save(path)

    overview = api.get("/api/overview", params={"source": "published"}).json()
    for molecule, backends in overview["by_molecule"].items():
        for backend in backends:
            first = save("/api/rank", {"molecule": molecule, "backend": backend}, "rank")
            for bond in (first or {}).get("bond_lengths", []):
                save(
                    "/api/rank",
                    {"molecule": molecule, "backend": backend, "bond_length": bond},
                    "rank",
                )
            save("/api/scan", {"molecule": molecule, "backend": backend}, "scan")

    for study in api.get("/api/replicated", params={"source": "published"}).json()["studies"]:
        save(
            "/api/replicated/recommendation",
            {"molecule": study["molecule"], "backend": study["backend"], "device": study["device"]},
        )

    runs = api.get("/api/runs", params={"source": "published"}).json()["runs"]
    (out_dir / "runs").mkdir(exist_ok=True)
    for row in runs:
        res = api.get(f"/api/runs/{row['hash']}", params={"source": "published"})
        if res.status_code == 200:
            (out_dir / "runs" / f"{row['hash']}.json").write_text(
                json.dumps(res.json(), separators=(",", ":")), encoding="utf-8"
            )
            counts["runs"] += 1
    return counts
