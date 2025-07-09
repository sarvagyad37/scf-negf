"""Li14 DFT + NEGF Solver

This package provides a first-principles implementation of a self-consistent
DFT + NEGF workflow for a 14-atom lithium chain.

Key sub-modules:
    geometry          – Build atomic coordinates and cell objects
    partition         – Generate device/lead partitions & coupling matrices
    self_energy       – Recursive surface Green-function & Σ(E) utilities
    scf               – Self-consistent DFT+NEGF driving routines
    postprocessing    – Transmission, DOS, and orbital visualization helpers

The project follows the protocols outlined in `docs/project-brain.md`.
"""

# Public re-exports for convenience
# (Only light-weight modules are re-imported here.)

__all__ = [
    "__version__",
    "self_energy",
    "visualize",
    "partition",
]

__version__ = "0.1.0"

# noqa import positions kept intentionally

from . import self_energy  # noqa: E402,F401
from . import visualize   # noqa: E402,F401
from . import partition   # noqa: E402,F401
from . import scf         # noqa: E402,F401
from . import mixing      # noqa: E402,F401
from . import alignment   # noqa: E402,F401
from . import kerker      # noqa: E402,F401
from . import diagnostics # noqa: E402,F401
from . import config      # noqa: E402,F401
from . import validation  # noqa: E402,F401
from . import postprocessing  # noqa: E402,F401
from . import lead_bulk   # noqa: E402,F401

__all__.append("scf")
__all__.append("mixing")
__all__.append("alignment")
__all__.append("kerker")
__all__.append("diagnostics")
__all__.append("config")
__all__.append("validation")
__all__.append("postprocessing")
__all__.append("lead_bulk") 