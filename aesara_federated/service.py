import asyncio
import logging
from typing import AsyncIterator, Callable, List, Optional, Sequence, Tuple, Type

import grpclib
import numpy as np
from betterproto import Message, ServiceStub
from grpclib.client import Channel, Stream
from grpclib.metadata import Deadline
from grpclib.stream import _RecvType, _SendType

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


def _compute_federated_logp(
    flop_input: FederatedLogpOpInput,
    perform_grad: LogpGradFunc,
) -> FederatedLogpOpOutput:
    """Wraps a ``perform_grad`` function with gRPC message decoding/encoding.

    Parameters
    ----------
    flop_input
        FederatedLogp input gRPC message.
    perform_grad
        The blackbox logp-gradient function.

    Returns
    -------
    flop_output
        FederatedLogp output gRPC message.
    """
    # Deserialize input arrays
    inputs = [ndarray_to_numpy(i) for i in flop_input.inputs]
    # Run the computation
    logp, gradients = perform_grad(*inputs)
    assert logp.shape == ()
    assert len(gradients) == len(inputs), f"inputs: {inputs}, gradients: {gradients}"
    # Encode results
    result = FederatedLogpOpOutput(
        log_potential=ndarray_from_numpy(logp),
        gradients=[ndarray_from_numpy(g) for g in gradients],
    )
    return result


class FederatedLogpOpService(FederatedLogpOpBase):
    """Implements the FederatedLogp service."""

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
        return _compute_federated_logp(federated_logp_op_input, self._perform_grad)

    async def evaluate_stream(
        self, federated_logp_op_input_iterator: AsyncIterator[FederatedLogpOpInput]
    ) -> AsyncIterator[FederatedLogpOpOutput]:
        _log.info("Evaluation stream opened")
        async for flop_input in federated_logp_op_input_iterator:
            yield _compute_federated_logp(flop_input, self._perform_grad)
        _log.info("Evaluation stream closed")


async def start_bidirectional_stream(
    *,
    client: ServiceStub,
    route: str,
    request_type: Type[Message],
    response_type: Type[Message],
    timeout: Optional[float] = None,
    deadline: Optional[Deadline] = None,
    metadata: Optional["MetadataLike"] = None,
) -> Stream[_SendType, _RecvType]:
    """Initializes a bidirectional message stream.

    Bidirectional gRPC streams are much faster
    than sending individual non-streamed calls.

    For parameter explanations see ``grpclib.client.Channel.request``.
    """
    # Open the stream
    stream = await client.channel.request(
        route,
        grpclib.const.Cardinality.STREAM_STREAM,
        request_type,
        response_type,
        timeout=client.timeout if timeout is None else timeout,
        deadline=client.deadline if deadline is None else deadline,
        metadata=client.metadata if metadata is None else metadata,
    ).__aenter__()
    # Send the request to start
    await stream.send_request()
    return stream


class FederatedLogpOpClient:
    """Client wrapper around the autogenerated gRPC client implementation."""

    def __init__(self, host: str, port: int) -> None:
        """Create a wrapper around the FederatedLogpOp gRPC client.

        Parameters
        ----------
        host : str
            IP address or host name of the remote gRPC server.
        port : int
            Port of the gRPC server.
        """
        self._channel = Channel(host, port)
        self._client = FederatedLogpOpStub(self._channel)
        self._loop = asyncio.get_event_loop()
        self._stream = self._loop.run_until_complete(
            start_bidirectional_stream(
                client=self._client,
                route="/FederatedLogpOp/EvaluateStream",
                request_type=FederatedLogpOpInput,
                response_type=FederatedLogpOpOutput,
            )
        )
        super().__init__()

    def __del__(self):
        # Announce stopping the streaming
        self._stream.end()
        # Close the stream
        self._loop.run_until_complete(self._stream.__aexit__())
        self._channel.close()
        return

    def evaluate(
        self, *inputs: Sequence[np.ndarray], use_stream=True
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Evaluate the federated blackbox logp gradient function on inputs.

        Parameters
        ----------
        *inputs
            NumPy `ndarray` inputs.
        use_stream : bool
            If ``True`` (default), the RPC is performed through a bidirectional stream,
            which is much faster than sending individual (unary/unary) RPCs.

        Returns
        -------
        logp : ndarray
            The log-potential of the federated blackbox function.
        *gradients
            Sequence of ``ndarray`` with gradients of ``logp`` w.r.t. ``inputs``.
        """
        # Encode inputs
        fpi = FederatedLogpOpInput(inputs=[ndarray_from_numpy(i) for i in inputs])

        # Make the asynchronous calls to the remote server
        if use_stream:
            eval_task = self._streamed_evaluate(fpi)
        else:
            eval_task = self._client.evaluate(fpi)
        fpo = self._loop.run_until_complete(eval_task)

        # Decode outputs
        logp = ndarray_to_numpy(fpo.log_potential)
        gradient = [ndarray_to_numpy(g) for g in fpo.gradients]
        return logp, gradient

    async def _streamed_evaluate(self, fpi: FederatedLogpOpInput) -> FederatedLogpOpOutput:
        """Internal wrapper around async methods of the bidirectional stream."""
        await self._stream.send_message(fpi)
        response = await self._stream.recv_message()
        assert response is not None
        return response
