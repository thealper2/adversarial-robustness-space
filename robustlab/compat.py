"""Small compatibility shims for the pinned library versions."""

import numpy as np

# ART 1.20.1 still calls ``np.asfarray`` (removed in NumPy 2.0) in the laser-beam and pixel attacks.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
