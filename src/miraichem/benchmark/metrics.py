"""Pure metric functions. All energies are total energies in Hartree."""

from __future__ import annotations

CHEMICAL_ACCURACY_HA = 1.6e-3  # 1.6 mHa, the usual target for useful chemistry


def abs_error_mha(e_vqe: float, e_exact: float) -> float:
    """Absolute energy error in milli-Hartree."""
    return abs(e_vqe - e_exact) * 1000.0


def within_chemical_accuracy(e_vqe: float, e_exact: float) -> bool:
    """True if the error is below 1.6 mHa."""
    return abs(e_vqe - e_exact) < CHEMICAL_ACCURACY_HA
