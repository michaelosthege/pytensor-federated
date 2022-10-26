"""Generic utility functions that have only external dependencies."""
import asyncio
import logging
from typing import Callable, Iterable, List, Optional, TypeVar

import nest_asyncio
import numpy as np

T = TypeVar("T")
_log = logging.getLogger(__file__)


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


def get_useful_event_loop() -> asyncio.AbstractEventLoop:
    """Like `asyncio.get_event_loop()` but actually useful.

    If the loop is already running, this function patches it using `nest_asyncio`,
    because otherwise it would be impossible to run something on it.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            _log.debug("Event loop is already running. Patching with nest_asyncio.")
            nest_asyncio.apply(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop
