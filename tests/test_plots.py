from miraichem.analysis.plots import plot_convergence, plot_pareto
from miraichem.benchmark.ranking import rank_results
from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult


def _run(h2, err, mitigated=False):
    cfg = RunConfig(molecule=h2, bond_length=0.735, backend="noisy", seed=int(err * 100))
    return RunResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        e_exact=-1.137,
        e_hf=-1.117,
        e_vqe=-1.137 + err / 1000,
        abs_error_mha=err,
        within_chemical_accuracy=err < 1.6,
        two_qubit_gates=10,
        total_shots=1000 + int(err),
        convergence_trace=[-1.0 + 0.004 * i for i in range(20)],
        convergence_trace_unmitigated=[-0.9 + 0.003 * i for i in range(20)] if mitigated else [],
    )


def test_convergence_plot_variants(h2):
    for as_error in (True, False):
        fig = plot_convergence(_run(h2, 5.0, mitigated=True), as_error=as_error)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "VQE energy" in labels and "unmitigated" in labels
    assert fig.axes[0].get_ylabel() == "Total energy (Hartree)"


def test_pareto_plot(h2):
    ranked = rank_results([_run(h2, 1.0), _run(h2, 20.0), _run(h2, 5.0)])
    for axis in ("shots", "gates"):
        fig = plot_pareto(ranked, axis)
        assert fig.axes[0].get_yscale() == "log"
        assert len(fig.axes[0].collections) >= 2
