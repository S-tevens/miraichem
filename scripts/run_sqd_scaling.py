"""Scaling study for SQD on N2: error and runtime against the number of qubits (20, 24, 28, 32).

Usage: python scripts/run_sqd_scaling.py [results-dir] [--seeds 5] [--molecules n2 n2_631g]

Every size has an exact reference (the 32-qubit one takes about 6 minutes once and is then saved in
results/exact, so it never runs again). Results are saved like any other SQD run and cached by
config hash, so an interrupted study resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.config import load_sqd_yaml
from miraichem.sqd.problem import build_problem
from miraichem.sqd.storage import load_sqd_results, save_sqd_result

ROOT = Path(__file__).resolve().parents[1]
MOLECULES = ["n2", "n2_631g_24q", "n2_631g_28q", "n2_631g"]
# (label, overrides). Same default as the 20-qubit seed study: noise 0.02, 3 rounds, 300 samples.
SETTINGS = [("1 layer", {}), ("2 layers", {"n_reps": 2})]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--molecules", nargs="+", default=MOLECULES)
    ap.add_argument(
        "--noise-model",
        default=None,
        help="fake IBM device (for example FakeFez) to drive the sampling noise instead of "
        "uniform bit flips",
    )
    ap.add_argument(
        "--samples-per-batch",
        type=int,
        default=300,
        help="bitstrings per subspace; 300 is the study default, larger values test whether the "
        "error at bigger sizes is limited by the subspace",
    )
    args = ap.parse_args()
    out = Path(args.results_dir)
    done = {r.config_hash for r in load_sqd_results(out) if r.status == "ok"}

    for name in args.molecules:
        mol, bond = load_sqd_yaml((ROOT / "configs" / "sqd" / f"{name}.yaml").read_text())
        print(f"== {mol.name}: building the problem and the exact energy...", flush=True)
        problem = build_problem(mol, bond, exact=True, cache_dir=out / "exact")
        print(
            f"   {problem.num_qubits} qubits, exact solve {problem.fci_time_s:.1f} s, "
            f"CCSD {problem.ccsd_time_s:.1f} s",
            flush=True,
        )
        for (label, extra), seed in itertools.product(SETTINGS, range(args.seeds)):
            cfg = SqdConfig(
                molecule=mol,
                bond_length=bond,
                readout_noise=0.0 if args.noise_model else 0.02,
                noise_model=args.noise_model,
                samples_per_batch=args.samples_per_batch,
                seed=seed,
                **extra,
            )
            if cfg.config_hash() in done:
                continue
            res = run_sqd(cfg, problem)
            save_sqd_result(res, out)
            err = (
                "failed: " + str(res.error_message)
                if res.status == "failed"
                else (f"{res.abs_error_mha:.2f} mHa in {res.wall_time_s:.0f} s")
            )
            print(f"   {label:9s} seed {seed}: {err}", flush=True)


if __name__ == "__main__":
    main()
