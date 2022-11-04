import argparse
import asyncio
import logging
import multiprocessing
import time
from typing import Sequence, Tuple

import aesara
import aesara.tensor as at
import grpclib
import numpy as np

from aesara_federated import ArraysToArraysService, wrap_logp_grad_func

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class LinearModelBlackbox:
    def __init__(
        self, data_x: np.ndarray, data_y: np.ndarray, sigma: float, delay: float = 0
    ) -> None:
        self._data_x = data_x
        self._data_y = data_y
        self._sigma = sigma
        self._delay = delay
        self._fn = self._make_function(data_x, data_y, sigma)
        super().__init__()

    @staticmethod
    def _make_function(x, y, sigma):
        intercept = at.scalar()
        slope = at.scalar()
        pred = intercept + x * slope

        pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * at.exp(-0.5 * ((y - pred) / sigma) ** 2)
        logp = at.log(pdf).sum()
        grad = at.grad(logp, wrt=[intercept, slope])
        fn = aesara.function(
            inputs=[intercept, slope],
            outputs=[logp, *grad],
        )
        return fn

    def __call__(
        self, *parameters: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        # This perform the computation and sleeps
        # until it took `self._delay` seconds.
        t0 = time.perf_counter()
        logp, *grads = self._fn(*parameters)
        t_elapsed = time.perf_counter() - t0
        time.sleep(max(0, self._delay - t_elapsed))
        return logp, grads


async def run_node_async(*, bind: str, port: int, delay: float):
    _log.info("Generating a secret dataset")
    x = np.linspace(0, 10, 10)
    sigma = 0.4
    y = np.random.RandomState(123).normal(1.5 + 0.5 * x, scale=sigma)

    import scipy.stats

    mle = scipy.stats.linregress(x, y)
    print(mle)

    _log.info("Compiling a model function")
    model_fn = LinearModelBlackbox(
        data_x=x,
        data_y=y,
        sigma=sigma,
        delay=delay,
    )
    _log.info("Binding the service to %s on port %i", bind, port)
    service = ArraysToArraysService(wrap_logp_grad_func(model_fn))
    server = grpclib.server.Server([service])
    await server.start(bind, port)
    await server.wait_closed()
    return


def run_node(bind_port_delay: Tuple[str, int, float]):
    try:
        bind, port, delay = bind_port_delay
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_node_async(bind=bind, port=port, delay=delay))
    except KeyboardInterrupt:

        class KeyboardInterruptError(Exception):
            pass

        # Re-raise as a real exception to populate to the parent proceess.
        raise KeyboardInterruptError()
    return


def run_node_pool(bind: str, ports: Sequence[int], delay: float):
    _log.info("Launching workers on %i subprocesses", len(ports))
    pool = multiprocessing.Pool(len(ports))
    try:
        pool.map(run_node, [(bind, p, delay) for p in ports])
    except KeyboardInterrupt:
        _log.info("Stopping workers...")
        pool.terminate()
        pool.join()
    _log.info("All workers exited.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a toy model as a worker node.")
    parser.add_argument(
        "--bind", default="0.0.0.0", help="IP address to run the ArraysToArrays gRPC service on."
    )
    parser.add_argument(
        "--ports",
        default=",".join(map(str, range(50000, 50015))),
        type=str,
        help="Port numbers for the ArraysToArrays gRPC service.",
    )
    parser.add_argument(
        "--delay",
        default=0,
        type=float,
        help="Seconds to sleep in each evaluation.",
    )
    args, _ = parser.parse_known_args()

    run_node_pool(
        bind=args.bind,
        ports=list(map(int, str(args.ports).split(","))),
        delay=args.delay,
    )
