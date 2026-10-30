"""Build the 3D structure data the web app shows: atoms, bonds and molecular-orbital surfaces.

Usage: python scripts/make_structures.py [output-dir]      (needs scikit-image: a one-off tool,
not a dependency of the package; the output is committed so the app never needs it)

H2, LiH, N2, O2 and NO get real orbitals: each is computed with PySCF (RHF; ROHF for the O2 triplet)
in the same STO-3G basis the benchmarks use, evaluated on a grid and turned into two surfaces
(positive and negative lobes of the orbital) with marching cubes.

FeS4 and Fe4S4 are drawn as IDEALISED structures only (a tetrahedral FeS4 and a cubane Fe4S4 core,
with typical Fe-S distances). They have NO computed orbitals: iron-sulphur clusters need active
spaces taken from the literature, which this project does not have, and the app says so.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from skimage import measure

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/published/structures")
ISO = 0.05  # orbital amplitude of the drawn surface
STEP = 0.22  # grid spacing in Angstrom
PAD = 3.4  # margin around the atoms in Angstrom
BOHR = 0.529177210903

MOLECULES = [
    # id, label, atoms (Angstrom), spin, note
    ("h2", "H₂", "H 0 0 0; H 0 0 0.735", 0, "STO-3G, bond 0.735 Å"),
    ("lih", "LiH", "Li 0 0 0; H 0 0 1.595", 0, "STO-3G, bond 1.595 Å"),
    ("n2", "N₂", "N 0 0 0; N 0 0 1.0977", 0, "STO-3G, bond 1.0977 Å"),
    ("o2", "O₂ (triplet)", "O 0 0 0; O 0 0 1.208", 2, "STO-3G, ROHF orbitals, bond 1.208 Å"),
    ("no", "NO (doublet)", "N 0 0 0; O 0 0 1.151", 1, "STO-3G, ROHF orbitals, bond 1.151 Å"),
]


def surfaces(
    mol, coeff_col: np.ndarray, grid: np.ndarray, shape: tuple[int, int, int], origin, step
):
    ao = mol.eval_gto("GTOval_sph", grid)
    psi = (ao @ coeff_col).reshape(shape)
    out = {}
    for sign, name in ((1, "pos"), (-1, "neg")):
        field = sign * psi
        if field.max() < ISO:
            out[name] = {"v": [], "f": []}
            continue
        verts, faces, _, _ = measure.marching_cubes(field, level=ISO, spacing=(step,) * 3)
        verts = verts + np.asarray(origin)
        out[name] = {
            "v": [round(float(c), 2) for c in verts.ravel()],
            "f": [int(i) for i in faces.ravel()],
        }
    return out


def orbitals_for(spec: str, spin: int):
    mol = gto.M(atom=spec, basis="sto3g", spin=spin, unit="Angstrom", verbose=0)
    mf = scf.ROHF(mol) if spin else scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    coords = mol.atom_coords() * BOHR
    lo = coords.min(axis=0) - PAD
    hi = coords.max(axis=0) + PAD
    axes = [np.arange(lo[k], hi[k] + STEP, STEP) for k in range(3)]
    shape = tuple(len(a) for a in axes)
    pts = np.array(list(product(*axes))) / BOHR  # PySCF grids are in Bohr
    out = []
    occ = mf.mo_occ
    n_occ = int((occ > 0).sum())
    for k in range(mol.nao):
        surf = surfaces(mol, mf.mo_coeff[:, k], pts, shape, lo, STEP)
        o = float(occ[k])
        tag = "HOMO" if k == n_occ - 1 else "LUMO" if k == n_occ else ""
        out.append(
            {"index": k + 1, "energy": float(mf.mo_energy[k]), "occupation": o, "tag": tag, **surf}
        )
    atoms = [
        [mol.atom_symbol(i), *[round(float(c), 3) for c in coords[i]]] for i in range(mol.natm)
    ]
    return atoms, out


def idealised_fes4():
    d = 2.30  # a typical Fe-S distance in tetrahedral FeS4 sites
    a = d / np.sqrt(3)
    s = [(a, a, a), (a, -a, -a), (-a, a, -a), (-a, -a, a)]
    return [["Fe", 0, 0, 0]] + [["S", *[round(float(c), 3) for c in p]] for p in s]


def idealised_fe4s4():
    d = 2.27  # a typical Fe-S distance in the cubane [4Fe-4S] core
    corners = list(product((-1, 1), repeat=3))
    atoms = []
    for c in corners:
        element = "Fe" if (c[0] * c[1] * c[2]) > 0 else "S"
        atoms.append([element, *[round(float(x * d / 2), 3) for x in c]])
    return atoms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for mid, label, spec, spin, note in MOLECULES:
        atoms, orbitals = orbitals_for(spec, spin)
        out.append(
            {
                "id": mid,
                "label": label,
                "kind": "computed",
                "note": note,
                "atoms": atoms,
                "orbitals": orbitals,
            }
        )
        print(mid, len(orbitals), "orbitals")
    out.append(
        {
            "id": "fes4",
            "label": "FeS₄",
            "kind": "schematic",
            "note": (
                "Idealised tetrahedral FeS₄ site (Fe-S 2.30 Å). "
                "Shape only: no orbitals were computed."
            ),
            "atoms": idealised_fes4(),
            "orbitals": [],
        }
    )
    out.append(
        {
            "id": "fe4s4",
            "label": "Fe₄S₄",
            "kind": "schematic",
            "note": (
                "Idealised cubane Fe₄S₄ core (Fe-S 2.27 Å). Shape only: no orbitals were computed."
            ),
            "atoms": idealised_fe4s4(),
            "orbitals": [],
        }
    )
    (OUT / "structures.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(
        "wrote",
        OUT / "structures.json",
        round((OUT / "structures.json").stat().st_size / 1024),
        "KB",
    )


if __name__ == "__main__":
    main()
