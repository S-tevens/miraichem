"""Control experiments for SQD on N2: what does the quantum circuit add?

Usage: python scripts/run_sqd_controls.py [results-dir] [--seeds 5] [--molecules n2 n2_631g_24q]

Two kinds of control, both measured against the exact energy:

1. The same SQD loop with the circuit left out. ``hf_noise`` samples only the Hartree-Fock state and
   adds the same bit-flip noise; ``uniform`` draws random bits. Both use the same recovery,
   post-selection and subspace solve as the circuit runs. They are saved as ordinary SQD results.
2. Subspaces of the same size as the ones the circuit runs found, chosen without a circuit: random
   strings, the lowest excitations from Hartree-Fock, and an oracle that keeps the strings carrying
   the most weight in the exact ground state (a bound no method can beat at that size). Their
   energies are written to ``<results-dir>/controls/<molecule>.json``, together with the error of
   every subspace size from 1 string up, which is the error-against-subspace-size curve.

The circuit runs are the ones already saved with the study's default settings (noise 0.02, 3
rounds, 300 samples per batch). Everything is cached, so an interrupted study resumes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.config import load_sqd_yaml
from miraichem.sqd.controls import equal_size, exact_weights
from miraichem.sqd.problem import build_problem
from miraichem.sqd.storage import load_sqd_results, save_sqd_result

ROOT = Path(__file__).resolve().parents[1]
MOLECULES = ["n2", "n2_631g_24q", "n2_631g_28q"]
GRID = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 130, 190, 250]
MHA = 1000.0


def is_study_run(r, name: str, bond: float) -> bool:
    c = r.config
    return (
        r.status == "ok"
        and c.molecule.name == name
        and abs(c.bond_length - bond) < 1e-9
        and c.sampler == "circuit"
        and c.noise_model is None
        and c.readout_noise == 0.02
        and c.iterations == 3
        and c.samples_per_batch == 300
        and c.num_batches == 3
        and c.shots == 20000
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--molecules", nargs="+", default=MOLECULES)
    ap.add_argument("--n-random", type=int, default=10, help="random subspaces per size")
    args = ap.parse_args()
    out = Path(args.results_dir)
    rng = np.random.default_rng(1234)

    for name in args.molecules:
        mol, bond = load_sqd_yaml((ROOT / "configs" / "sqd" / f"{name}.yaml").read_text())
        print(f"== {mol.name}: building the problem...", flush=True)
        problem = build_problem(mol, bond, exact=True, cache_dir=ROOT / "results" / "exact")

        # 1. the SQD loop with the circuit replaced
        done = {r.config_hash for r in load_sqd_results(out) if r.status == "ok"}
        for sampler in ("hf_noise", "uniform"):
            for seed in range(args.seeds):
                cfg = SqdConfig(
                    molecule=mol, bond_length=bond, readout_noise=0.02, sampler=sampler, seed=seed
                )
                if cfg.config_hash() in done:
                    continue
                res = run_sqd(cfg, problem)
                save_sqd_result(res, out)
                msg = res.error_message if res.status == "failed" else f"{res.abs_error_mha:.2f} mHa"
                print(f"   {sampler:9s} seed {seed}: {msg}", flush=True)

        # 2. subspaces of the same size, chosen without a circuit
        print("   exact ground-state weights...", flush=True)
        weights, e_exact = exact_weights(problem)
        assert abs(e_exact - problem.e_exact) < 1e-6, "exact weights disagree with the exact energy"
        n_strings = len(weights)

        def errors(k: int) -> dict:
            e = equal_size(problem, k, weights, rng, args.n_random)
            rand = [(x - e_exact) * MHA for x in e["random"]]
            return {
                "random_mean_mha": float(np.mean(rand)),
                "random_std_mha": float(np.std(rand, ddof=1)) if len(rand) > 1 else 0.0,
                "excitation_mha": (e["excitation"] - e_exact) * MHA,
                "oracle_mha": (e["oracle"] - e_exact) * MHA,
            }

        curve = []
        for k in [g for g in GRID if g <= n_strings]:
            curve.append({"k": k, "dim": k * k, **errors(k)})
            print(f"   size {k:4d}: oracle {curve[-1]['oracle_mha']:.3f} mHa", flush=True)

        runs = [
            r
            for r in load_sqd_results(out)
            if is_study_run(r, mol.name, bond) and r.config.seed < args.seeds
        ]
        sqd = []
        for r in sorted(runs, key=lambda r: (r.config.n_reps, r.config.seed)):
            k = round(math.sqrt(r.subspace_dim))
            sqd.append(
                {
                    "seed": r.config.seed,
                    "n_reps": r.config.n_reps,
                    "k": k,
                    "dim": r.subspace_dim,
                    "error_mha": r.abs_error_mha,
                    **errors(min(k, n_strings)),
                }
            )
            print(
                f"   SQD {r.config.n_reps} layer(s) seed {r.config.seed}: "
                f"{r.abs_error_mha:.2f} mHa on {k} strings",
                flush=True,
            )

        path = out / "controls" / f"{mol.name.lower()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "molecule": mol.name,
                    "qubits": problem.num_qubits,
                    "bond_length": bond,
                    "n_strings": n_strings,
                    "nelec": list(problem.nelec),
                    "e_exact": e_exact,
                    "e_hf": problem.e_hf,
                    "e_ccsd": problem.e_ccsd,
                    "hf_error_mha": abs(problem.e_hf - e_exact) * MHA,
                    "ccsd_error_mha": abs(problem.e_ccsd - e_exact) * MHA,
                    "n_random": args.n_random,
                    "curve": curve,
                    "sqd": sqd,
                },
                indent=2,
            )
        )
        print(f"   saved {path}", flush=True)


if __name__ == "__main__":
    main()
