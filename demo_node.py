import asyncio
import logging
from typing import List, Sequence, Tuple

import aesara
import aesara.tensor as at
import grpclib
import numpy as np

from aesara_federated import FederatedLogpOpService
from aesara_federated.service import LogpGradFunc

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class LinearModelBlackbox:
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, sigma: float) -> None:
        self._data_x = data_x
        self._data_y = data_y
        self._sigma = sigma
        self._fn = self._make_function(data_x, data_y, sigma)
        super().__init__()

    @staticmethod
    def _make_function(x, y, sigma) -> LogpGradFunc:
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

    def __call__(self, *parameters: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        logp, *grads = self._fn(*parameters)
        return logp, grads


async def run_node(port: int = 50051):
    _log.info("Generating a secret dataset")
    x = np.linspace(0, 10, 10)
    sigma = 0.4
    y = np.random.normal(1.5 + 0.5 * x, scale=sigma)

    import scipy.stats

    mle = scipy.stats.linregress(x, y)
    print(mle)

    _log.info("Compiling a model function")
    model_fn = LinearModelBlackbox(
        data_x=x,
        data_y=y,
        sigma=sigma,
    )
    _log.info("Starting the service on port %i", port)
    service = FederatedLogpOpService(model_fn)
    server = grpclib.server.Server([service])
    await server.start("127.0.0.1", port)
    await server.wait_closed()
    return


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_node())
