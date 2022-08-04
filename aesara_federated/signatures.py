"""
This module contains only type and signature definitions.
"""
from typing import Callable, Sequence, Tuple

import numpy as np

ComputeFunc = Callable[
    [Sequence[np.ndarray]],  # arbitrary number of input arrays
    Sequence[np.ndarray],  # arbitrary number of output arrays
]
"""
Signature of a generic compute function
taking multiple NumPy arrays as inputs
and returning multiple NumPy arrays as outputs.
"""

LogpFunc = Callable[
    [Sequence[np.ndarray]],  # arbitrary number of input arrays
    np.ndarray,  # scalar log-p
]
"""
Signature of a log-probability function without gradients.
For example a log-likelihood function.
"""

LogpGradFunc = Callable[
    [Sequence[np.ndarray]],  # arbitrary number of input arrays
    Tuple[np.ndarray, Sequence[np.ndarray]],  # scalar log-p and gradients w.r.t. each input
]
"""
Signature of a log-probability function with gradients w.r.t. inputs.
"""
