"""Ranking, Pareto fronts and recommendation, on small synthetic data (no simulation)."""

import pytest
from typer.testing import CliRunner

from miraichem import cli
from miraichem.benchmark.ranking import (
    Weights,
    gate_cost,
    pareto_front,
    rank_results,
    ranking_dataframe,
    recommend,
)
from miraichem.benchmark.storage import save_result
from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult

_counter = iter(range(10_000))


def make(h2, err, gates=None, shots=0, backend="noisy", status="ok", **cfg):
    """A synthetic result with the given error (mHa), two-qubit gates and shots."""
    config = RunConfig(
        molecule=h2, bond_length=0.735, backend=backend, seed=next(_counter), **cfg
    )  # unique seed -> unique hash
    ok = status == "ok"
    return RunResult(
        config=config,
        config_hash=config.config_hash(),
        status=status,
        e_exact=-1.137,
        e_hf=-1.117,
        e_vqe=-1.137 + err / 1000 if ok else None,
        abs_error_mha=err if ok else None,
        within_chemical_accuracy=err < 1.6 if ok else None,
        two_qubit_gates=gates,
        logical_depth=40,
        total_shots=shots,
        n_function_evals=50,
    )


def test_pareto_front_basic():
    pts = [(1, 10), (2, 5), (3, 1), (2, 8), (4, 4), (1, 10)]
    # (2,8) dominated by (2,5); (4,4) dominated by (3,1); duplicate (1,10) stays on the front
    assert pareto_front(pts) == [0, 1, 2, 5]
    assert pareto_front([]) == []
    assert pareto_front([(1, 1)]) == [0]


def test_weights_validation_and_normalisation():
    with pytest.raises(ValueError):
        Weights(-1, 1, 1)
    with pytest.raises(ValueError):
        Weights(0, 0, 0)
    w = Weights(2, 1, 1).normalized()
    assert (w.accuracy, w.gates, w.shots) == pytest.approx((0.5, 0.25, 0.25))


def test_accuracy_dominates_by_default_and_weights_can_flip(h2):
    accurate_costly = make(h2, 5.0, gates=100, shots=1_000_000)
    cheap_inaccurate = make(h2, 50.0, gates=10, shots=100_000)
    ranked = rank_results([cheap_inaccurate, accurate_costly])
    assert ranked[0].result is accurate_costly and ranked[0].rank == 1
    cost_first = rank_results([cheap_inaccurate, accurate_costly], Weights(0.05, 0.5, 0.45))
    assert cost_first[0].result is cheap_inaccurate


def test_scores_are_bounded_and_constant_metrics_are_safe(h2):
    runs = [make(h2, e, gates=20, shots=1000) for e in (1.0, 10.0, 100.0)]  # same cost everywhere
    ranked = rank_results(runs)
    assert all(0.0 <= x.score <= 1.0 for x in ranked)
    assert [x.error_mha for x in ranked] == [1.0, 10.0, 100.0]
    assert rank_results([]) == []


def test_pareto_flags(h2):
    a = make(h2, 1.0, gates=100, shots=500)  # most accurate
    b = make(h2, 10.0, gates=10, shots=100)  # cheapest
    c = make(h2, 20.0, gates=200, shots=900)  # worse than a on every axis
    ranked = {x.result.config_hash: x for x in rank_results([a, b, c])}
    assert ranked[a.config_hash].on_front_shots and ranked[a.config_hash].on_front_gates
    assert ranked[b.config_hash].on_front_shots and ranked[b.config_hash].on_front_gates
    assert not ranked[c.config_hash].on_front_shots
    assert not ranked[c.config_hash].on_front_gates


def test_recommends_cheapest_within_chemical_accuracy(h2):
    exact_costly = make(h2, 0.01, gates=200, shots=2_000_000)
    ok_cheap = make(h2, 1.2, gates=20, shots=200_000)  # chemically accurate and cheap
    cheaper_but_off = make(h2, 30.0, gates=3, shots=10_000)
    rec = recommend([exact_costly, ok_cheap, cheaper_but_off], "H2", "noisy")
    assert rec.best.result is ok_cheap
    assert rec.reached_chemical_accuracy and rec.n_candidates == 3
    text = rec.justification
    assert "1.2 mHa" in text and "20 two-qubit gates" in text and "200,000 shots" in text
    assert "cheapest" in text and "2 of 3" in text
    assert "10.0x the shots" in text and "10.0x the gates" in text  # vs the lowest-error run


def test_recommendation_when_nothing_reaches_chemical_accuracy(h2):
    a = make(h2, 80.0, gates=50, shots=300_000)
    b = make(h2, 200.0, gates=5, shots=50_000)
    rec = recommend([a, b], "h2", "noisy")
    assert not rec.reached_chemical_accuracy
    assert rec.best.result is a  # best score overall
    assert "None of the 2 runs reached chemical accuracy" in rec.justification


def test_filters_failed_other_backend_and_molecule(h2):
    good = make(h2, 5.0, gates=10, shots=100)
    failed = make(h2, 1.0, status="failed")
    ideal = make(h2, 0.0, backend="ideal")
    rec = recommend([good, failed, ideal], "h2", "noisy")
    assert rec.n_candidates == 1 and rec.best.result is good
    with pytest.raises(ValueError, match="No successful runs"):
        recommend([good], "lih", "noisy")
    with pytest.raises(ValueError, match="No successful runs"):
        recommend([failed], "h2", "noisy")


def test_ideal_runs_use_logical_depth_as_gate_cost(h2):
    ideal = make(h2, 0.0, backend="ideal")
    assert ideal.two_qubit_gates is None and gate_cost(ideal) == 40
    rec = recommend([ideal, make(h2, 0.5, backend="ideal")], "h2", "ideal")
    assert rec.best.within_chemical_accuracy


def test_most_common_bond_length_is_default(h2):
    near = [make(h2, 5.0, gates=10, shots=10) for _ in range(2)]
    far = make(h2, 1.0, gates=10, shots=10)
    far.config.bond_length = 2.0
    assert recommend(near + [far], "h2", "noisy").n_candidates == 2
    assert recommend(near + [far], "h2", "noisy", bond_length=2.0).best.result is far


def test_ranking_dataframe(h2):
    ranked = rank_results([make(h2, 5.0, gates=10, shots=100), make(h2, 50.0, gates=5, shots=50)])
    df = ranking_dataframe(ranked)
    assert list(df["rank"]) == [1, 2]
    assert {"error_mHa", "two_qubit_gates", "pareto_shots", "config_hash"} <= set(df.columns)


def test_cli_rank(h2, tmp_path):
    for r in (make(h2, 1.2, gates=20, shots=200_000), make(h2, 30.0, gates=3, shots=10_000)):
        save_result(r, tmp_path)
    args = ["rank", "--results-dir", str(tmp_path)]
    out = CliRunner().invoke(cli.app, [*args, "--molecule", "h2"])
    assert out.exit_code == 0, out.output
    assert "Recommended for h2" in out.output and "leaderboard" in out.output
    bad = CliRunner().invoke(cli.app, [*args, "--molecule", "lih"])
    assert bad.exit_code == 1
