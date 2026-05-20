"""Generic utility functions that have only external dependencies."""

import asyncio
import logging
from typing import Callable, Iterable, List, Optional, TypeVar

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
    """Get a running/current/new event loop, in that order of priority."""
    loop: asyncio.AbstractEventLoop
    try:
        # First try to get an already running event loop.
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # There is no running loop.
        try:
            # Is there a current, not running loop?
            loop = asyncio.get_event_loop()
        except:
            # There's no current loop, so we must create one.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    return loop
