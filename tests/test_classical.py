"""Reference energies must match well-known values (catches unit/geometry mistakes)."""

import pytest

from miraichem.chemistry.classical import compute_reference

H2_EQ = 0.735  # Angstrom
LIH_EQ = 1.595


def test_h2_reference_values(h2):
    ref = compute_reference(h2, H2_EQ)
    assert ref.e_hf == pytest.approx(-1.117, abs=1e-3)
    assert ref.e_exact == pytest.approx(-1.137, abs=1e-3)
    assert ref.exact_method == "FCI"
    assert ref.e_exact < ref.e_hf  # correlation energy is negative
    assert ref.e_nuclear_repulsion == pytest.approx(0.7199, abs=1e-3)


def test_h2_qubits_expected(h2):
    assert compute_reference(h2, H2_EQ).n_qubits_expected == 4


def test_lih_active_space_reference(lih):
    ref = compute_reference(lih, LIH_EQ)
    assert ref.exact_method == "CASCI"
    assert ref.n_qubits_expected == 6
    # Total energies (core + nuclear included): HF about -7.862, CASCI slightly below it.
    assert ref.e_hf == pytest.approx(-7.862, abs=2e-3)
    assert ref.e_hf - ref.e_exact == pytest.approx(0.019, abs=3e-3)
