"""Lead self-energy utilities wrapping self_energy.surface_gf_* helpers.

This module sits one abstraction level above ``self_energy.py``:
– It loads the principal-layer (H0, H1, S0, S1) blocks that define a semi-infinite
  periodic lead (usually produced by ``lead_bulk.py``)
– Computes the retarded self-energy Σ(E) on a user-supplied energy grid
– Optionally persists results to disk (``.npz``) and reloads them on demand so
  expensive surface-GF recursions are executed once per run.

The public API intentionally mirrors TranSIESTA / ATK semantics to make later
integration with the SCF loop straightforward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any

import json
import time

import numpy as np

from .self_energy import self_energy_grid

__all__ = [
    "load_lead_blocks",
    "compute_sigma_grid",
]


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_lead_blocks(json_file: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read *lead_data.json* produced by ``lead_bulk.py``.

    The JSON schema is simple; only the four blocks needed for Σ are required.
    Keys:
        H00, H01   – complex Hamiltonian blocks (lists of lists)
        S00, S01   – overlap blocks (may be omitted for orthogonal bases)

    Returns
    -------
    (H0, H1, S0, S1)  with *dtype complex128*  (S0/S1 are ``None`` if absent)
    """
    with Path(json_file).expanduser().open() as fh:
        data: Dict[str, Any] = json.load(fh)

    H0 = np.asarray(data["H00"], dtype=np.complex128)
    H1 = np.asarray(data["H01"], dtype=np.complex128)

    S0 = np.asarray(data["S00"], dtype=np.complex128) if "S00" in data else None
    S1 = np.asarray(data["S01"], dtype=np.complex128) if "S01" in data else None

    return H0, H1, S0, S1


# -----------------------------------------------------------------------------
# Σ(E) grid computation with on-disk cache
# -----------------------------------------------------------------------------

def _default_cache_path(json_file: str | Path, *, bias: float | None = None) -> Path:
    """Return cache path; includes bias value when supplied (task 4.7b)."""

    jf = Path(json_file).with_suffix("")
    if bias is None or abs(bias) < 1e-12:
        return Path(f"{jf}_sigma_cache.npz")

    # encode bias in mV to keep filename readable
    bias_mV = int(round(bias * 27.2114 * 1000))  # Ha → eV → mV
    return Path(f"{jf}_sigma_bias{bias_mV:+d}mV.npz")


def compute_sigma_grid(
    energies: Iterable[float],
    *,
    lead_json: str | Path,
    eta: float = 1e-6,
    potential_shift: float = 0.0,
    bias: float = 0.0,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    shift_tol: float = 1e-3,  # Ha (≈ 1 meV) – if |Δshift| > tol, recompute
) -> np.ndarray:
    """Compute Σ(E) for all *energies* and return a 3-D array (N_E, N_orb, N_orb).

    Parameters
    ----------
    energies
        Real energy grid (Ha).
    lead_json
        Path to *lead_data.json* containing PL Hamiltonian + overlap blocks.
    eta
        Broadening added to *g* (defaults to 1 µHa).
    potential_shift
        Potential shift to apply to the lead.
    bias
        Bias to apply to the lead.
    cache_path / overwrite
        If *cache_path* is given (or derived automatically) the function will
        try to load/store an ``.npz`` file with keys ``E``, ``Sigma``, and ``shift``.
    shift_tol
        Tolerance for potential shift recomputation.
    """
    energies_arr = np.asarray(list(energies))  # can be real or complex

    cache_file = Path(cache_path) if cache_path else _default_cache_path(lead_json, bias=bias)

    if not overwrite and cache_file.exists():
        saved = np.load(cache_file, allow_pickle=False)

        e_saved = saved["E"]
        grid_ok = e_saved.shape == energies_arr.shape and np.allclose(e_saved, energies_arr)
        if "shift" in saved:
            shift_ok = abs(float(saved["shift"]) - potential_shift) < shift_tol
        else:
            shift_ok = potential_shift < shift_tol

        if "bias" in saved:
            bias_ok = abs(float(saved["bias"]) - bias) < 1e-12
        else:
            bias_ok = abs(bias) < 1e-12

        if grid_ok and shift_ok and bias_ok:
            return saved["Sigma"]
        # Otherwise fall through and recompute

    H0, H1, S0, S1 = load_lead_blocks(lead_json)

    # ------------------------------------------------------------------
    # Apply uniform potential shift by evaluating Σ at (E − Δ) which is
    # algebraically equivalent to adding Δ to all on-site terms of H.
    # This keeps the bulk-lead reference consistent with the extended
    # device Hamiltonian that already includes the same Δ applied to its
    # diagonal.
    # ------------------------------------------------------------------
    energies_shifted = energies_arr - potential_shift

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Compute Σ(E) in one shot – caller must ensure the Hamiltonian / overlap
    # are well conditioned; no automatic retries or fallbacks are attempted.
    # ------------------------------------------------------------------

    _, Sigma = self_energy_grid(
        energies_shifted,
        H0,
        H1,
        S0=S0,
        S1=S1,
        eta=eta,
    )

    dt = time.perf_counter() - t0
    print(
        f"[lead_self_energy] Σ grid computed for {len(energies_arr)} E points in {dt:.2f}s → {cache_file.name}"
    )

    # persist
    np.savez_compressed(cache_file, E=energies_arr, Sigma=Sigma, shift=potential_shift, bias=bias)

    return Sigma 