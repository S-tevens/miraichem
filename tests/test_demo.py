"""The offline demo snapshot: file names (mirrored in the frontend) and a real export."""

import json
from pathlib import Path

import pytest

from miraichem.api.demo import demo_key, export_demo

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published"

# The same literals are asserted in frontend/src/lib/lib.test.ts. If one side changes, both fail.
KEY_EXAMPLES = [
    ("/api/overview", {"source": "published", "n": "3"}, "_api_overview.json"),
    (
        "/api/rank",
        {
            "molecule": "H2",
            "backend": "noisy",
            "bond_length": "0.735",
            "w_acc": "70",
            "source": "published",
        },
        "_api_rank_backend=noisy_bond_length=0.735_molecule=H2.json",
    ),
    (
        "/api/rank",
        {"molecule": "H2", "backend": "ideal", "bond_length": "1.0"},
        "_api_rank_backend=ideal_bond_length=1_molecule=H2.json",
    ),
    (
        "/api/rank",
        {"molecule": "LiH", "backend": "noisy"},
        "_api_rank_backend=noisy_molecule=LiH.json",
    ),
    ("/api/sqd", {"source": "published", "n": "2"}, "_api_sqd.json"),
    ("/api/sqd/options", {}, "_api_sqd_options.json"),
    (
        "/api/replicated/recommendation",
        {"molecule": "H2", "backend": "noisy", "device": "FakeFez", "reliability": "0.5"},
        "_api_replicated_recommendation_backend=noisy_device=FakeFez_molecule=H2.json",
    ),
    ("/api/trust/index", {"source": "published"}, "_api_trust_index.json"),
    (
        "/api/trust",
        {"kind": "vqe", "molecule": "H2", "backend": "noisy", "device": "FakeFez", "n": "2"},
        "_api_trust_backend=noisy_device=FakeFez_kind=vqe_molecule=H2.json",
    ),
    (
        "/api/trust",
        {"kind": "vqe", "molecule": "LiH", "backend": "ideal", "device": ""},
        "_api_trust_backend=ideal_kind=vqe_molecule=LiH.json",
    ),
]


@pytest.mark.parametrize(("path", "query", "expected"), KEY_EXAMPLES)
def test_demo_keys_match_the_documented_examples(path, query, expected):
    assert demo_key(path, query) == expected


def test_scoring_weights_do_not_change_the_file_name():
    base = {"molecule": "H2", "backend": "noisy"}
    assert demo_key("/api/rank", {**base, "w_acc": "10"}) == demo_key("/api/rank", base)


@pytest.mark.skipif(
    not (PUBLISHED / "summary" / "replication_runs.csv").is_file(), reason="no published results"
)
def test_export_writes_every_view_the_app_asks_for(tmp_path):
    out = tmp_path / "demo"
    counts = export_demo(out, PUBLISHED, ROOT / "configs")
    assert counts["runs"] > 100 and counts["rank"] >= 2
    for name in (
        "_api_sources.json",
        "_api_options.json",
        "_api_overview.json",
        "_api_replicated.json",
        "_api_devices.json",
        "_api_hardware.json",
        "_api_curves.json",
        "_api_predictor.json",
        "_api_sqd.json",
        "_api_sqd_options.json",
        "_api_structures.json",
        "_api_trust_index.json",
        "_api_rank_backend=noisy_molecule=H2.json",
    ):
        assert (out / name).is_file(), name
    # the default source in the snapshot is the published one, and every file is strict JSON
    assert json.loads((out / "_api_sources.json").read_text())["default"] == "published"
    for f in out.rglob("*.json"):
        json.loads(f.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    # a run opened from a leaderboard row exists in the snapshot
    rank = json.loads((out / "_api_rank_backend=noisy_molecule=H2.json").read_text())
    assert (out / "runs" / f"{rank['best_hash']}.json").is_file()
    # every trust report the page can ask for is in the snapshot
    subjects = json.loads((out / "_api_trust_index.json").read_text())["subjects"]
    assert subjects
    for subject in subjects:
        query = {k: subject[k] for k in ("kind", "molecule", "backend", "device")}
        assert (out / demo_key("/api/trust", query)).is_file(), query
