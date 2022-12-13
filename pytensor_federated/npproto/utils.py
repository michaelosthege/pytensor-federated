"""
Helper functions such as converters between ``ndarray`` and ``Ndarray``.
"""
import numpy

from . import Ndarray


def ndarray_from_numpy(arr: numpy.ndarray) -> Ndarray:
    return Ndarray(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        data=bytes(arr.data),
        strides=list(arr.strides),
    )


def ndarray_to_numpy(nda: Ndarray) -> numpy.ndarray:
    return numpy.ndarray(
        buffer=nda.data,
        shape=nda.shape,
        dtype=numpy.dtype(nda.dtype),
        strides=nda.strides,
    )
