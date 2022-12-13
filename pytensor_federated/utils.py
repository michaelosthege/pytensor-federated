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

    Otherwise, the currently idle, or a new event loop is returned.

    NOTE: This function achieves the same as running `nest_asyncio.apply()` once,
          and using `asyncio.get_event_loop()`, but it has fewer side-effects.
    """
    # First try to get an already running event loop.
    loop = asyncio._get_running_loop()
    if loop is not None:
        # We're already in a coroutine and `loop` is currently
        # `await`ing the code that called this function.
        # This loop can't `await` something new, but must be
        # patched to support running a new loop inside.
        if not hasattr(loop, "_nest_patched"):
            _log.debug("Event loop is already running. Patching with nest_asyncio.")
            nest_asyncio.apply(loop)
    else:
        # There is no currently active loop, so we must create a new one.
        loop = asyncio.get_event_loop()
    return loop
