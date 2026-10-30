"""Stretched-bond study for SQD on N2: where classical CCSD breaks down, does SQD hold?

Usage: python scripts/run_sqd_stretch.py [results-dir] [--seeds 5] [--molecules n2 n2_631g_24q]

N2 has a triple bond, which is the textbook case where coupled cluster fails as the atoms move
apart (the correlation becomes strong). Every size here still has an exact reference, so each SQD
and CCSD error is measured against the exact energy. The settings are the same at every bond length
and match the equilibrium seed study (noise 0.02, 3 rounds, 300 samples per batch), so the points
line up with it. Results are saved like any other SQD run and cached by config hash, so an
interrupted study resumes where it stopped.
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
MOLECULES = ["n2", "n2_631g_24q"]
STRETCHED = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]  # Angstrom; the equilibrium point already exists
SETTINGS = [("1 layer", {}), ("2 layers", {"n_reps": 2})]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--molecules", nargs="+", default=MOLECULES)
    ap.add_argument("--bonds", nargs="+", type=float, default=STRETCHED)
    args = ap.parse_args()
    out = Path(args.results_dir)
    done = {r.config_hash for r in load_sqd_results(out) if r.status == "ok"}

    for name in args.molecules:
        mol, _ = load_sqd_yaml((ROOT / "configs" / "sqd" / f"{name}.yaml").read_text())
        for bond in args.bonds:
            print(f"== {mol.name} at {bond} A: building the problem...", flush=True)
            try:
                problem = build_problem(mol, bond, exact=True, cache_dir=ROOT / "results" / "exact")
            except Exception as exc:  # noqa: BLE001 - a point that will not converge is skipped
                print(f"   skipped: {type(exc).__name__}: {exc}", flush=True)
                continue
            hf = abs(problem.e_hf - problem.e_exact) * 1000
            cc = abs(problem.e_ccsd - problem.e_exact) * 1000
            print(f"   {problem.num_qubits} qubits: HF {hf:.1f} mHa, CCSD {cc:.1f} mHa", flush=True)
            for (label, extra), seed in itertools.product(SETTINGS, range(args.seeds)):
                cfg = SqdConfig(
                    molecule=mol,
                    bond_length=bond,
                    readout_noise=0.02,
                    seed=seed,
                    **extra,
                )
                if cfg.config_hash() in done:
                    continue
                res = run_sqd(cfg, problem)
                save_sqd_result(res, out)
                msg = (
                    "failed: " + str(res.error_message)
                    if res.status == "failed"
                    else f"{res.abs_error_mha:.2f} mHa, {100 * res.subspace_fraction:.0f}% of space"
                )
                print(f"   {label:9s} seed {seed}: {msg}", flush=True)


if __name__ == "__main__":
    main()
