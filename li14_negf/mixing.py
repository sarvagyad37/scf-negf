"""Mixing utilities (DIIS / linear) for SCF loops (Task 4.4/4.5).

Only density-matrix–agnostic linear algebra is implemented so the mixer can be
used for densities, potentials, or Fock matrices alike.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np

__all__ = ["MixerDIIS"]


class MixerDIIS:
    """Pulay/Anderson mixer with optional Kerker preconditioning (Task 4.5b)."""

    def __init__(self, beta: float = 0.1, size: int = 8, precond: callable | None = None):
        if size < 2:
            raise ValueError("DIIS history size must be ≥ 2")
        self.beta = beta
        self.size = size
        self._x_hist: Deque[np.ndarray] = deque(maxlen=size)
        self._r_hist: Deque[np.ndarray] = deque(maxlen=size)

        # Optional preconditioner (e.g. Kerker) that acts on residual matrices
        self._precond = precond

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Return mixed variable given current *x_in* and new output *x_out*."""

        res_mat = x_out - x_in

        if self._precond is not None:
            res_mat = self._precond(res_mat)

        r = res_mat.ravel()  # 1-D view after preconditioning

        # Store current vectors
        self._x_hist.append(x_out.copy())
        self._r_hist.append(r.copy())

        # If not enough history, do simple linear mixing
        if len(self._r_hist) < 2:
            return x_in + self.beta * (x_out - x_in)

        try:
            c = self._solve_coeffs()  # mixing coefficients
            x_new = np.sum([c[i] * self._x_hist[i] for i in range(len(c))], axis=0)
            return x_new
        except np.linalg.LinAlgError:
            # Fall back to linear mixing on singular matrix
            return x_in + self.beta * (x_out - x_in)

    def reset(self) -> None:
        """Clear stored history so the next update falls back to linear mixing."""
        self._x_hist.clear()
        self._r_hist.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_coeffs(self) -> np.ndarray:
        """Solve Pulay equation B c = -1 with ∑c_i = 1 constraint."""

        m = len(self._r_hist)
        B = np.empty((m + 1, m + 1))
        B[-1, :] = B[:, -1] = -1.0
        B[-1, -1] = 0.0

        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(self._r_hist[i], self._r_hist[j])

        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0

        coeffs = np.linalg.solve(B, rhs)[:-1]  # last element is lagrange mult.
        return coeffs 