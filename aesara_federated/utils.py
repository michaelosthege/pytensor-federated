"""Generic utility functions that have only external dependencies."""
from typing import Callable, Iterable, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def argmin_none_or_func(
    items: Iterable[Optional[T]],
    func: Callable[[T], float],
) -> Optional[int]:
    """Argmin over the return values of `func` while ignoring `None` items.

    Returns `None` if all items are `None`.

    Parameters
    ----------
    items
        Iterable of objects.
    func
        A callable that returns a float when applied to non-`None` elements of `items`.
    """
    items = list(items)
    if not any(i is not None for i in items):
        return None

    values: List[float] = [(np.inf if item is None else func(item)) for item in items]

    return np.argmin(values)
