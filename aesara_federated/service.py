import asyncio
import logging
import os
import threading
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import grpclib
import numpy as np
import psutil
from betterproto import Message, ServiceStub
from grpclib.client import Channel, Stream
from grpclib.metadata import Deadline
from grpclib.stream import _RecvType, _SendType

from .npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from .rpc import (
    ArraysToArraysServiceBase,
    ArraysToArraysServiceStub,
    GetLoadParams,
    GetLoadResult,
    InputArrays,
    OutputArrays,
)
from .signatures import ComputeFunc
from .utils import argmin_none_or_func

if TYPE_CHECKING:
    from betterproto.grpc.grpclib_client import MetadataLike


_log = logging.getLogger(__file__)


def _run_compute_func(
    func_input: InputArrays,
    func: ComputeFunc,
) -> OutputArrays:
    """Wraps a ``compute_func`` with gRPC message decoding/encoding.

    Parameters
    ----------
    func_input
        InputArrays gRPC message.
    func
        The blackbox function.

    Returns
    -------
    func_output
        OutputArrays gRPC message.
    """
    # Deserialize input arrays
    inputs = [ndarray_to_numpy(i) for i in func_input.items]
    # Run the computation
    outputs = func(*inputs)
    # Encode results
    result = OutputArrays(
        items=[ndarray_from_numpy(np.asarray(o)) for o in outputs],
        uuid=func_input.uuid,
    )
    return result


class ArraysToArraysService(ArraysToArraysServiceBase):
    """Implements a gRPC service around a ``ComputeFunc``."""

    def __init__(
        self,
        compute_func: ComputeFunc,
    ) -> None:
        self._compute_func = compute_func
        self._n_clients = 0
        # Start monitoring of CPU load average
        self.determine_load()
        super().__init__()

    def determine_load(self) -> GetLoadResult:
        """Determines the current system load."""
        # We're only interested in the 1-minute average
        load_1, _, _ = psutil.getloadavg()
        return GetLoadResult(
            n_clients=self._n_clients,
            percent_cpu=load_1 / psutil.cpu_count() * 100,
            percent_ram=psutil.virtual_memory().percent,
        )

    async def evaluate(
        self,
        input_arrays: InputArrays,
    ) -> OutputArrays:
        return _run_compute_func(input_arrays, self._compute_func)

    async def evaluate_stream(
        self, input_arrays_iterator: AsyncIterator[InputArrays]
    ) -> AsyncIterator[OutputArrays]:
        _log.info("Evaluation stream opened")
        self._n_clients += 1
        async for input in input_arrays_iterator:
            yield _run_compute_func(input, self._compute_func)
        _log.info("Evaluation stream closed")
        self._n_clients -= 1

    async def get_load(self, get_load_params: GetLoadParams) -> GetLoadResult:
        return self.determine_load()


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


async def _streamed_evaluate(
    input: InputArrays, stream: Stream[_SendType, _RecvType]
) -> OutputArrays:
    """Internal wrapper around async methods of the bidirectional stream."""
    await stream.send_message(input)
    response = await stream.recv_message()
    if response is None:
        raise Exception("Received unexpected `None` response.")
    return response


async def get_loads_async(
    hosts_and_ports: Sequence[Tuple[str, int]], *, timeout: float = 2
) -> Sequence[Optional[GetLoadResult]]:
    """Retrieve load information from all servers that respond within a timeout.

    Parameters
    ----------
    hosts_and_ports : list of (host, port) tuples, optional
        List of hostnames and port number of ArraysToArrays gRPC servers.

    Returns
    -------
    loads
        ``GetLoadResult`` objects or ``None`` for each server.
    """
    # Asynchronously get the current load information for each server
    load_tasks: List[Coroutine[Any, Any, GetLoadResult]] = []
    for host, port in hosts_and_ports:
        channel = Channel(host, port)
        client = ArraysToArraysServiceStub(channel)
        load_tasks.append(client.get_load(GetLoadParams(), timeout=timeout))

    # Complete the asynchronous load queries
    loads: List[Optional[GetLoadResult]] = []
    for load_task in load_tasks:
        try:
            loads.append(await load_task)
        except (ConnectionRefusedError, asyncio.exceptions.TimeoutError):
            loads.append(None)

    return loads


class ClientPrivates:
    """Collects gRPC objects that are private to one client instance."""

    def __init__(
        self,
        channel: Channel,
        client: ArraysToArraysServiceStub,
        stream: Stream[_SendType, _RecvType],
    ):
        self.channel = channel
        self.client = client
        self.stream = stream

    @staticmethod
    async def connect(host: str, port: int):
        channel = Channel(host, port)
        client = ArraysToArraysServiceStub(channel)
        stream = await start_bidirectional_stream(
            client=client,
            route="/ArraysToArraysService/EvaluateStream",
            request_type=InputArrays,
            response_type=OutputArrays,
        )
        return ClientPrivates(channel, client, stream)

    @staticmethod
    async def connect_balanced(hosts_and_ports: Sequence[Tuple[str, int]]):
        # First shuffle the list. This achieves two things:
        # 1. Our queries are send in random order
        # 2. If two servers have identical load, a random one will be selected.
        hosts_and_ports = np.random.permutation(hosts_and_ports)

        # Asynchronously retriev load information from all servers
        loads = await get_loads_async(hosts_and_ports)

        # Find one with minimal number of clients
        isel = argmin_none_or_func(loads, lambda l: l.n_clients)
        if isel is None:
            raise TimeoutError(
                f"None of {len(hosts_and_ports)} servers responded to load information requests."
            )

        host, port = hosts_and_ports[isel]
        return await ClientPrivates.connect(host, port)


_privates: Dict[str, ClientPrivates] = {}
"""
This dictionary contains the non-resusable client connections, indexed by
identifiers that must be instance-, thread- and process-specific (see `thread_pid_id`).
"""


def thread_pid_id(obj: object) -> str:
    """A process- and thread- specific identifier of an object."""
    return f"{id(obj)}-{os.getpid()}-{threading.get_ident()}"


class ArraysToArraysServiceClient:
    """Wraps the autogenerated gRPC client implementation with a ``ComputeFunc`` signature."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        *,
        hosts_and_ports: Sequence[Tuple[str, int]] = None,
    ) -> None:
        """Create a wrapper around the ArraysToArraysOp gRPC client.

        Parameters
        ----------
        host : str
            IP address or host name of the remote gRPC server.
        port : int
            Port of the gRPC server.
        hosts_and_ports : list of (host, port) tuples, optional
            If provided, this takes precedence over `host` and `port`.
            Uses client-side load balancing to choose which server to connec to.
        """
        self._host = host
        self._port = port
        self._hosts_and_ports = hosts_and_ports
        # The gRPC objects must not be pickled and are stored in _privates.
        # They will be initialized on first access.
        super().__init__()

    def __del__(self):
        _id = thread_pid_id(self)
        priv = _privates.get(_id, None)
        if priv is not None:
            _log.info("Closing evaluation stream")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(priv.stream.end())
            priv.channel.close()
            # Remove from the dict, otherwise it won't be garbage-collected
            del _privates[_id]
        return

    def __call__(self, *inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Alias for ``.evaluate(*inputs)``."""
        return self.evaluate(*inputs)

    def evaluate(self, *inputs: Sequence[np.ndarray], use_stream=True) -> Sequence[np.ndarray]:
        """Evaluate the federated compute function on inputs.

        Parameters
        ----------
        *inputs
            NumPy `ndarray` inputs.
        use_stream : bool
            If ``True`` (default), the RPC is performed through a bidirectional stream,
            which is much faster than sending individual (unary/unary) RPCs.

        Returns
        -------
        *outputs
            Sequence of ``ndarray``s returned by the federated compute function.
        """
        loop = asyncio.get_event_loop()

        # Encode inputs
        input = InputArrays(
            items=[ndarray_from_numpy(np.asarray(i)) for i in inputs], uuid=str(uuid.uuid4())
        )

        # Get a handle on this instance's private gRPC objects
        _id = thread_pid_id(self)
        if not _id in _privates:
            _log.debug(f"Connecting client {_id}")
            if self._hosts_and_ports:
                connect_task = ClientPrivates.connect_balanced(self._hosts_and_ports)
            else:
                connect_task = ClientPrivates.connect(self._host, self._port)
            _privates[_id] = loop.run_until_complete(connect_task)
        priv = _privates[_id]

        # Make the asynchronous calls to the remote server
        if use_stream:
            eval_task = _streamed_evaluate(input, priv.stream)
        else:
            eval_task = priv.client.evaluate(input)
        output = loop.run_until_complete(eval_task)
        if output.uuid != input.uuid:
            raise Exception("Response does not correspond to the request.")

        # Decode outputs
        outputs = [ndarray_to_numpy(o) for o in output.items]
        return outputs
