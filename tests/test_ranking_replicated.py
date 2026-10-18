import pandas as pd
import pytest

from miraichem.benchmark.ranking_replicated import (
    recommend_replicated,
    replicated_configs,
    wilson_interval,
)


def _runs(name: str, errors: list[float], gates: float, shots: float = 1000, **over) -> list[dict]:
    base = {
        "molecule": "H2",
        "bond_length": 0.735,
        "mapping": "parity",
        "ansatz": name,
        "reps": 1,
        "optimizer": "spsa",
        "maxiter": 150,
        "backend": "noisy",
        "device": "FakeSherbrooke",
        "shots": 4096,
        "mitigation": "none",
        "zne": "",
        "n_evals": 10,
        "two_qubit_gates": gates,
        "logical_depth": 5,
        "total_shots": shots,
    }
    base.update(over)
    return [{**base, "seed": i, "error_mHa": e} for i, e in enumerate(errors)]


def _df(*groups) -> pd.DataFrame:
    return pd.DataFrame([row for g in groups for row in g])


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.30
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.70 < lo < 0.75
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_reliable_configs_pick_the_cheapest_reliable_one():
    df = _df(
        _runs("uccsd", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], gates=50),
        _runs("hea", [0.4, 0.5, 0.6, 0.7, 0.9, 1.2], gates=3),
        _runs("bad", [30, 31, 32, 33, 34, 35], gates=1),
    )
    rec = recommend_replicated(df, "H2", "noisy")
    assert rec.reliable
    assert rec.best.key["ansatz"] == "hea"  # reliable and cheaper than uccsd
    assert "cheapest of the 2 configurations" in rec.justification


def test_unreliable_picks_cheapest_among_statistically_tied():
    df = _df(
        _runs("accurate", [3.0, 3.5, 4.0, 4.5, 5.0, 3.2], gates=40),
        _runs("close_cheap", [3.4, 3.9, 4.4, 4.9, 5.4, 3.6], gates=2),
    )
    rec = recommend_replicated(df, "H2", "noisy")
    assert not rec.reliable
    assert rec.lowest_mean.key["ansatz"] == "accurate"
    assert rec.best.key["ansatz"] == "close_cheap"
    assert "none is reliably chemically accurate" in rec.justification


def test_huge_variance_cheap_config_is_not_considered_tied():
    # mean 150 +- 300: "not significantly worse" by t-test, but practically far worse
    noisy = [2.0, 800.0, 3.0, 4.0, 90.0, 10.0]
    df = _df(
        _runs("good", [3.0, 3.5, 4.0, 4.5, 5.0, 3.2], gates=40),
        _runs("wild", noisy, gates=1),
    )
    rec = recommend_replicated(df, "H2", "noisy")
    assert rec.best.key["ansatz"] == "good"


def test_configs_with_too_few_seeds_are_skipped():
    df = _df(_runs("a", [1.0, 2.0], gates=1), _runs("b", [1.0, 2.0, 3.0], gates=2))
    configs = replicated_configs(df, "H2", "noisy", min_seeds=3)
    assert [c.key["ansatz"] for c in configs] == ["b"]
    with pytest.raises(ValueError):
        recommend_replicated(df, "H2", "noisy", min_seeds=5)


def test_different_devices_are_different_configurations():
    df = _df(
        _runs("hea", [1.0, 2.0, 3.0], gates=1, device="FakeSherbrooke"),
        _runs("hea", [5.0, 6.0, 7.0], gates=1, device="FakeBrisbane"),
    )
    assert len(replicated_configs(df, "H2", "noisy")) == 2


def test_old_csv_without_cost_columns_still_works():
    df = _df(_runs("hea", [1.0, 2.0, 3.0], gates=1)).drop(
        columns=["device", "zne", "two_qubit_gates", "logical_depth", "total_shots"]
    )
    rec = recommend_replicated(df, "H2", "noisy")
    assert rec.best.n_seeds == 3


def test_compare_devices_rank_correlation_and_recommendations():
    from miraichem.benchmark.devices import compare_devices, summarize_agreement

    rows = []
    for device, scale in (("FakeA", 1.0), ("FakeB", 1.3)):
        for mit, base in (("none", 40.0), ("readout", 15.0), ("readout+zne", 4.0)):
            errs = [base * scale * f for f in (0.8, 0.9, 1.0, 1.1, 1.2, 1.0)]
            rows += _runs("uccsd", errs, gates=4, mitigation=mit, device=device)
    # a configuration measured on one device only must not distort the comparison
    rows += _runs("hea", [1.0, 2.0, 3.0, 4.0, 5.0], gates=1, device="FakeA", reps=9)
    cmp = compare_devices(pd.DataFrame(rows), "H2", "noisy")
    assert cmp.devices == ["FakeA", "FakeB"] and cmp.n_configs == 3
    assert cmp.spearman.loc["FakeA", "FakeB"] == pytest.approx(1.0)
    assert {r.best.key["mitigation"] for r in cmp.recommendations.values()} == {"readout+zne"}
    assert "same on all devices" in summarize_agreement(cmp)


def test_compare_devices_needs_two_devices():
    from miraichem.benchmark.devices import compare_devices

    df = _df(_runs("uccsd", [1.0, 2.0, 3.0, 4.0, 5.0], gates=4, device="FakeA"))
    with pytest.raises(ValueError):
        compare_devices(df, "H2", "noisy")
