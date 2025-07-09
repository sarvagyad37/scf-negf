"""Project-wide default parameters (Task 4.8a – kwargs exposure).

A minimal *runtime* configuration mechanism using a plain dictionary.  This is
*not* intended as a full-blown settings system – it merely centralises the
important numerical knobs so users can override them programmatically without
editing each function call.

Example
-------
>>> import li14_negf as ln
>>> ln.config.set_param("mixing_beta", 0.2)
>>> mixer = ln.mixing.MixerDIIS(beta=ln.config.get_param("mixing_beta"))
"""

from __future__ import annotations

from typing import Any, Dict

_defaults: Dict[str, Any] = {
    # Contour integration
    "contour_radius": 5.0,      # Ha
    "contour_n_poles": 8,
    "realaxis_window": 5.0,     # Ha
    "realaxis_n_points": 16,
    # Numerical damping / broadening
    "eta": 1e-6,               # Ha
    "smear_sigma": 0.001,      # Ha  (≈ 300 K)
    # Mixing controls
    "mixing_beta": 0.2,  # slightly more aggressive linear weight
    "diis_size": 4,      # smaller DIIS subspace for stability
    # Kerker preconditioner strength (0<α≤1)
    "kerker_alpha": 0.5,
}


def get_param(name: str):
    """Return current value of *name* (raises *KeyError* if unknown)."""
    return _defaults[name]


def set_param(name: str, value: Any) -> None:
    """Override parameter *name* at runtime (must exist)."""
    if name not in _defaults:
        raise KeyError(f"Unknown parameter '{name}'")
    _defaults[name] = value 