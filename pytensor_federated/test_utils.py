import asyncio

from pytensor_federated import utils
from pytensor_federated.rpc import GetLoadResult


def test_argmin_load():
    assert utils.argmin_none_or_func([], float) is None
    assert utils.argmin_none_or_func([None, None], float) is None
    loads = [
        None,
        GetLoadResult(3, 0.5, 0.2),  # <- min RAM
        None,
        GetLoadResult(2, 0.05, 0.4),  # <- min CPU
        GetLoadResult(1, 0.1, 0.6),  # <- min n_clients
    ]
    assert utils.argmin_none_or_func(loads, lambda l: l.percent_ram) == 1
    assert utils.argmin_none_or_func(loads, lambda l: l.percent_cpu) == 3
    assert utils.argmin_none_or_func(loads, lambda l: l.n_clients) == 4
    pass


def test_get_useful_event_loop():
    # The test is not running in an event loop
    assert asyncio._get_running_loop() is None

    # This gives a new, non-running loop
    loop = utils.get_useful_event_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    assert not loop.is_running()

    # Inside a coroutine the loop is marked as running,
    # because the following code is waiting for the coroutine.
    async def check_is_running():
        assert loop.is_running()

    loop.run_until_complete(check_is_running())

    # Calling `get_useful_event_loop` inside the coroutine
    # should patch the already-running loop to support reentrance.
    async def check_nesting():
        assert loop.is_running()
        nloop = utils.get_useful_event_loop()
        assert hasattr(nloop, "_nest_patched")
        loop.run_until_complete(asyncio.sleep(0.01))

    loop.run_until_complete(check_nesting())
    pass
