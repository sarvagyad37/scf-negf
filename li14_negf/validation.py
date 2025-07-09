"""Post-SCF validation helpers (Task 4.9).

Includes flatness check of on-site potential profile, transmission sanity, and
archiving convenience.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "potential_flatness",
    "transmission_consistency",
    "archive_run",
]

_eV_per_Ha = 27.2114


def potential_flatness(
    potential: np.ndarray,
    outer_indices: Sequence[int],
    *,
    tol_meV: float = 50.0,
) -> Tuple[bool, float]:
    """Return (*is_flat*, max|ΔV|) over *outer_indices* of *potential* (Ha).

    Parameters
    ----------
    potential
        1-D array of on-site potentials (Hartree).
    outer_indices
        Indices corresponding to outermost principal-layer atoms on both sides.
    tol_meV
        Tolerance in milli-electron-volts for maximum variation.
    """

    vals = potential[list(outer_indices)]
    spread_Ha = float(np.max(vals) - np.min(vals))
    spread_meV = spread_Ha * _eV_per_Ha * 1e3
    return spread_meV <= tol_meV, spread_meV


def transmission_consistency(
    T_EF: float,
    n_open: int,
    *,
    tol: float = 1e-2,
) -> bool:
    """Check |T(E_F) − N_open| ≤ tol."""

    return abs(T_EF - float(n_open)) <= tol


def archive_run(out_dir: str | Path = "outputs", *, stamp: str | None = None) -> Path:
    """Create timestamped archive directory and return its Path."""

    base = Path(out_dir).expanduser().absolute()
    base.mkdir(parents=True, exist_ok=True)

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_path = base / f"run_{stamp}"
    run_path.mkdir()
    return run_path 