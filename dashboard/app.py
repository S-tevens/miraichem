"""MiraiChem dashboard: explore saved benchmark results (no recomputation).

Run:  streamlit run dashboard/app.py
Reads only saved results, so it works offline and is safe for a live demo. Two sources are offered
in the sidebar: the team's published results (``results/published/``, committed to the repository)
and your own local runs (``results/``; override the folder with the sidebar box or the
MIRAICHEM_RESULTS_DIR environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from miraichem.analysis.dissociation import plot_curve_plotly
from miraichem.benchmark.ranking import (
    Weights,
    rank_results,
    ranking_dataframe,
    recommend,
    select_runs,
)
from miraichem.benchmark.ranking_scan import recommend_across_scan
from miraichem.dashboard_helpers import (
    PIPELINE_DOT,
    before_after_figure,
    bond_lengths_for,
    convergence_figure,
    filter_runs,
    hardware_figure,
    hardware_table,
    has_run_results,
    load_dashboard_data,
    mitigation_figure,
    mitigation_summary,
    pareto_figure,
    run_label,
)

st.set_page_config(page_title="MiraiChem", page_icon=":atom_symbol:", layout="wide")

TAB_NAMES = [
    "Overview",
    "Leaderboard",
    "Convergence",
    "Dissociation curve",
    "Mitigation",
    "Real hardware",
    "Recommendation",
]


def _stamp(results_dir: Path) -> float:
    """Latest modification time of any result file, so the cache refreshes when results change."""
    files = list(results_dir.rglob("*.json")) if results_dir.exists() else []
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner="Loading saved results...")
def _load(results_dir: str, stamp: float):  # noqa: ARG001 - stamp only invalidates the cache
    return load_dashboard_data(Path(results_dir))


def _empty(message: str, hint: str | None = None) -> None:
    st.info(message)
    if hint:
        st.code(hint, language="bash")


# --- sidebar ---

st.sidebar.title("MiraiChem")
st.sidebar.caption("Benchmarking VQE configurations on noisy IBM quantum backends")
local_dir = Path(
    st.sidebar.text_input("Results folder", os.environ.get("MIRAICHEM_RESULTS_DIR", "results"))
)
sources: dict[str, Path] = {}
if has_run_results(local_dir / "published"):
    sources["Published results (full team runs)"] = local_dir / "published"
if has_run_results(local_dir):
    sources["My local runs"] = local_dir
if len(sources) > 1:
    results_dir = sources[st.sidebar.radio("Data source", list(sources))]
elif sources:
    (label, results_dir), *_ = sources.items()
    st.sidebar.caption(f"Data source: {label.lower()}")
else:
    results_dir = local_dir
if st.sidebar.button("Reload results"):
    st.cache_data.clear()
data = _load(str(results_dir), _stamp(results_dir))
for warning in data.warnings:
    st.sidebar.warning(warning)

st.title("MiraiChem")
st.caption(
    "Which VQE setup is the most reliable for a small molecule on a noisy IBM backend? "
    "All numbers below come from saved runs; energies are total energies in Hartree."
)

if not data.ok_results:
    _empty(
        f"No saved results found in `{results_dir}`. Run a sweep first, then reload.",
        "miraichem sweep configs/sweeps/quick_h2.yaml --max-workers 4",
    )
    st.stop()

molecule = st.sidebar.selectbox("Molecule", data.molecules)
backends = data.backends_for(molecule)
backend = st.sidebar.selectbox(
    "Backend",
    backends,
    index=backends.index("noisy") if "noisy" in backends else 0,
    format_func=lambda b: {"ideal": "ideal simulator", "noisy": "noisy simulator"}.get(b, b),
)
bonds = bond_lengths_for(data.ok_results, molecule, backend)
bond_length = st.sidebar.selectbox(
    "Bond length (A)", bonds, help="Leaderboard, convergence and mitigation use this geometry."
)

runs_here = select_runs(data.results, molecule, backend, bond_length)

tabs = st.tabs(TAB_NAMES)

# --- overview ---

with tabs[0]:
    st.subheader("What this does")
    st.write(
        "VQE results depend on setup choices (ansatz, optimizer, qubit mapping, error mitigation, "
        "shots). MiraiChem sweeps those choices, scores each run against the exact answer, and "
        "recommends the most reliable and cheapest configuration, including a check on real "
        "IBM hardware."
    )
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    ok = data.ok_results
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved runs", len(ok))
    c2.metric("Molecules", len(data.molecules))
    c3.metric("Within chemical accuracy", sum(bool(r.within_chemical_accuracy) for r in ok))
    c4.metric("Real-hardware jobs", sum(len(h.evaluations) for h in data.hardware))
    st.caption(
        "Chemical accuracy = error below 1.6 mHa. We claim an engineering contribution "
        "(a reproducible, cost-aware benchmark), not new chemistry."
    )

# --- leaderboard ---

with tabs[1]:
    st.subheader(f"{molecule} leaderboard: {backend} backend at {bond_length} A")
    f1, f2, f3, f4 = st.columns(4)
    ansatz_opts = sorted({r.config.ansatz for r in runs_here})
    mapping_opts = sorted({r.config.mapping for r in runs_here})
    opt_opts = sorted({r.config.optimizer for r in runs_here})
    mit_opts = sorted({r.config.mitigation for r in runs_here})
    pick_ansatz = f1.multiselect("Ansatz", ansatz_opts, default=ansatz_opts)
    pick_mapping = f2.multiselect("Mapping", mapping_opts, default=mapping_opts)
    pick_opt = f3.multiselect("Optimizer", opt_opts, default=opt_opts)
    pick_mit = f4.multiselect("Mitigation", mit_opts, default=mit_opts)
    with st.expander("Score weights (accuracy matters most by default)"):
        w1, w2, w3 = st.columns(3)
        wa = w1.slider("Accuracy", 0.0, 1.0, 0.7, 0.05)
        wg = w2.slider("Two-qubit gates", 0.0, 1.0, 0.2, 0.05)
        ws = w3.slider("Shots", 0.0, 1.0, 0.1, 0.05)
    only_ca = st.checkbox("Only show runs within chemical accuracy")

    # An emptied filter box means "nothing selected", not "no restriction".
    if pick_ansatz and pick_mapping and pick_opt and pick_mit:
        chosen = filter_runs(runs_here, pick_ansatz, pick_mapping, pick_opt, pick_mit)
    else:
        chosen = []
    if not chosen:
        _empty("No runs match these filters.")
    elif wa + wg + ws == 0:
        _empty("Set at least one weight above zero.")
    else:
        ranked = rank_results(chosen, Weights(wa, wg, ws))
        shown = [x for x in ranked if x.within_chemical_accuracy] if only_ca else ranked
        df = ranking_dataframe(shown).drop(columns=["config_hash"])
        st.caption(
            f"{len(shown)} of {len(ranked)} runs. Green rows reach chemical accuracy. "
            "Score: 0 = best, 1 = worst (lower is better)."
        )
        styled = df.style.apply(
            lambda row: (
                ["background-color: rgba(27,158,119,0.18)" if row["chemical_accuracy"] else ""]
                * len(row)
            ),
            axis=1,
        )
        st.dataframe(styled, width="stretch", hide_index=True, height=360)
        st.download_button(
            "Download this table (CSV)",
            df.to_csv(index=False),
            f"{molecule}_{backend}_leaderboard.csv",
        )
        axis = st.radio("Cost axis", ["shots", "gates"], horizontal=True)
        st.plotly_chart(pareto_figure(ranked, axis), width="stretch", key=f"pareto_{axis}")

# --- convergence ---

with tabs[2]:
    st.subheader("Convergence of one run")
    ranked_all = rank_results(runs_here)
    if not ranked_all:
        _empty("No runs for this selection.")
    else:
        labels = {f"#{x.rank}  {x.label}  ({x.error_mha:.3g} mHa)": x for x in ranked_all[:60]}
        pick = st.selectbox("Run (best first)", list(labels))
        run = labels[pick].result
        view = st.radio("Show", ["Energy", "Error (log scale)"], horizontal=True)
        st.plotly_chart(
            convergence_figure(run, as_error=view != "Energy"), width="stretch", key="conv"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final energy (Ha)", f"{run.e_vqe:.6f}")
        m2.metric("Error (mHa)", f"{run.abs_error_mha:.3g}")
        m3.metric("Evaluations", run.n_function_evals)
        m4.metric("Parameters", run.num_params)

# --- dissociation curve ---

with tabs[3]:
    st.subheader("Dissociation curve: energy vs bond length")
    curves = [c for c in data.curves if c.molecule.lower() == molecule.lower()]
    if not curves:
        _empty(
            "No dissociation curves saved for this molecule yet.",
            f"miraichem curve --molecule configs/molecules/{molecule.lower()}.yaml --backend ideal",
        )
    else:
        names = {f"{c.label}  [{c.backend}]": c for c in curves}
        curve = names[st.selectbox("Curve", list(names))]
        st.plotly_chart(plot_curve_plotly(curve), width="stretch", key="curve")
        worst = curve.max_vqe_error_mha
        c1, c2, c3 = st.columns(3)
        c1.metric("Bond lengths", len(curve.bond_lengths))
        c2.metric("Within chemical accuracy", f"{curve.fraction_chemically_accurate:.0%}")
        c3.metric("Worst VQE error (mHa)", f"{worst:.3g}" if worst is not None else "-")
        st.caption("Hartree-Fock fails as the bond stretches; a good VQE follows the exact curve.")

# --- mitigation ---

with tabs[4]:
    st.subheader("Effect of error mitigation")
    if backend == "ideal":
        _empty("Mitigation applies to the noisy backend. Pick it in the sidebar.")
    else:
        fig = mitigation_figure(runs_here)
        if fig is None:
            _empty("Not enough runs with different mitigation settings at this geometry.")
        else:
            st.plotly_chart(fig, width="stretch", key="mit_bars")
            med = mitigation_summary(runs_here)
            cols = st.columns(len(med))
            for col, (mode, value) in zip(cols, med.items(), strict=True):
                col.metric(f"Median error: {mode}", f"{value:.3g} mHa")
            st.caption(
                "One seed per configuration, so small differences are within shot noise. "
                "Mitigation matters most with the Parity mapping (fewer, shallower circuits)."
            )
        scatter = before_after_figure(
            [
                r
                for r in data.ok_results
                if r.config.molecule.name == molecule and r.config.backend == backend
            ]
        )
        if scatter is not None:
            st.plotly_chart(scatter, width="stretch", key="mit_scatter")

# --- real hardware ---

with tabs[5]:
    st.subheader("Real IBM quantum hardware")
    hw = [h for h in data.hardware if h.molecule.lower() == molecule.lower()]
    if not hw:
        _empty(
            "No real-hardware results saved for this molecule. The dashboard works without them.",
            f"miraichem hardware --molecule configs/molecules/{molecule.lower()}.yaml "
            "--config-hash <hash> --allow-hardware",
        )
    else:
        st.warning(
            "These numbers come from a REAL quantum device (fixed-parameter evaluation of the best "
            "simulator parameters). Error bars are statistical standard errors of one run."
        )
        for i, h in enumerate(hw):
            st.markdown(
                f"**{h.molecule}** at {h.bond_length} A on `{h.backend}`, {h.shots} shots "
                f"(source simulator run `{h.source_config_hash}`)"
            )
            left, right = st.columns([3, 2])
            left.plotly_chart(hardware_figure(h), width="stretch", key=f"hw_fig_{i}")
            right.dataframe(hardware_table(h), width="stretch", hide_index=True)

# --- recommendation ---

with tabs[6]:
    st.subheader("Recommended configuration")
    try:
        scan = recommend_across_scan(data.results, molecule, backend)
        st.markdown("**Across a bond-length scan** (the more reliable recommendation)")
        st.success(scan.justification)
        cfg = scan.best.result.config
        st.markdown("Reproduce it:")
        st.code(
            f"miraichem curve --molecule configs/molecules/{molecule.lower()}.yaml "
            f"--config-hash {scan.best.result.config_hash} --backend {backend}",
            language="bash",
        )
        st.caption(
            f"{cfg.ansatz} ansatz (reps {cfg.ansatz_reps}), {cfg.optimizer}, "
            f"{cfg.mapping} mapping, "
            f"mitigation: {cfg.mitigation}."
        )
    except ValueError:
        st.info(
            "No configuration has been scanned over several bond lengths yet, so only the "
            "single-geometry recommendation below is available."
        )
    try:
        rec = recommend(data.results, molecule, backend, Weights(), bond_length)
        st.markdown(f"**At one geometry** ({bond_length} A): {run_label(rec.best.result)}")
        st.info(rec.justification)
        best = rec.best.result
        st.code(
            f"miraichem run --molecule configs/molecules/{molecule.lower()}.yaml "
            f"--bond-length {best.config.bond_length} --mapping {best.config.mapping} "
            f"--ansatz {best.config.ansatz} --optimizer {best.config.optimizer} "
            f"--backend {best.config.backend} --mitigation {best.config.mitigation} "
            f"--seed {best.config.seed}",
            language="bash",
        )
    except ValueError as exc:
        _empty(str(exc))
