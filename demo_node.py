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
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, sigma: float, delay: float=0) -> None:
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
        logp, *grads = self._fn(*parameters)
        # Simulate expensive compute
        time.sleep(self._delay)
        return logp, grads


async def run_node_async(*, bind: str, port: int):
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
    )
    _log.info("Binding the service to %s on port %i", bind, port)
    service = ArraysToArraysService(wrap_logp_grad_func(model_fn))
    server = grpclib.server.Server([service])
    await server.start(bind, port)
    await server.wait_closed()
    return


def run_node(bind_port: Tuple[str, int]):
    try:
        bind, port = bind_port
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_node_async(bind=bind, port=port))
    except KeyboardInterrupt:

        class KeyboardInterruptError(Exception):
            pass

        # Re-raise as a real exception to populate to the parent proceess.
        raise KeyboardInterruptError()
    return


def run_node_pool(ports: Sequence[int]):
    _log.info("Launching workers on %i subprocesses", len(ports))
    pool = multiprocessing.Pool(len(ports))
    try:
        pool.map(run_node, [(args.bind, p) for p in ports])
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
    args, _ = parser.parse_known_args()

    run_node_pool(list(map(int, str(args.ports).split(","))))
