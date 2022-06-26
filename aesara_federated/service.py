import asyncio
import logging
from typing import Callable, List, Sequence, Tuple

import numpy as np
from grpclib.client import Channel

from .npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from .rpc import (
    FederatedLogpOpBase,
    FederatedLogpOpInput,
    FederatedLogpOpOutput,
    FederatedLogpOpStub,
)

_log = logging.getLogger(__file__)


LogpGradFunc = Callable[
    [Sequence[np.ndarray]],  # arbitrary number of input arrays
    Tuple[np.ndarray, List[np.ndarray]],  # scalar log-p and gradients w.r.t. each input
]


class FederatedLogpOpService(FederatedLogpOpBase):
    def __init__(
        self,
        perform_grad: LogpGradFunc,
    ) -> None:
        self._perform_grad = perform_grad
        super().__init__()

    async def evaluate(
        self,
        federated_logp_op_input: FederatedLogpOpInput,
    ) -> FederatedLogpOpOutput:
        # Deserialize input arrays
        inputs = [ndarray_to_numpy(i) for i in federated_logp_op_input.inputs]
        # Run the computation
        logp, gradients = self._perform_grad(*inputs)
        assert logp.shape == ()
        assert len(gradients) == len(inputs), f"inputs: {inputs}, gradients: {gradients}"
        # Encode results
        result = FederatedLogpOpOutput(
            log_potential=ndarray_from_numpy(logp),
            gradients=[ndarray_from_numpy(g) for g in gradients],
        )
        return result


class FederatedLogpOpClient:
    def __init__(self, host: str, port: int) -> None:
        self._channel = Channel(host, port)
        self._client = FederatedLogpOpStub(self._channel)
        super().__init__()

    def __del__(self):
        self._channel.close()
        return

    def evaluate(self, *inputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        fpi = FederatedLogpOpInput(inputs=[ndarray_from_numpy(i) for i in inputs])
        eval_task = self._client.evaluate(fpi)
        loop = asyncio.get_event_loop()
        fpo = loop.run_until_complete(eval_task)
        logp = ndarray_to_numpy(fpo.log_potential)
        gradient = [ndarray_to_numpy(g) for g in fpo.gradients]
        return logp, gradient
