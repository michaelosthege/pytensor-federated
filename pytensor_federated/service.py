import asyncio
import logging
import os
import random
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
from .utils import argmin_none_or_func, get_useful_event_loop

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


async def get_load_async(host: str, port: int, timeout: float = 5) -> Optional[GetLoadResult]:
    """Retrieve load information from a server.

    Parameters
    ----------
    host : str
        IP address or host name of the remote gRPC server.
    port : int
        Port of the gRPC server.
    timeout : float
        Seconds to wait for a response.

    Returns
    -------
    load
        ``GetLoadResult`` object
        or ``None`` if the server did not respond in a timely manner.
    """
    channel = Channel(host, port)
    client = ArraysToArraysServiceStub(channel)
    try:
        load = await client.get_load(GetLoadParams(), timeout=timeout)
    except (ConnectionRefusedError, asyncio.exceptions.TimeoutError, OSError):
        load = None
    channel.close()
    return load


async def get_loads_async(
    hosts_and_ports: Sequence[Tuple[str, int]], *, timeout: float = 5
) -> Sequence[Optional[GetLoadResult]]:
    """Retrieve load information from all servers that respond within a timeout.

    Parameters
    ----------
    hosts_and_ports : list of (host, port) tuples, optional
        List of hostnames and port number of ArraysToArrays gRPC servers.
    timeout : float
        Seconds to wait for a response.

    Returns
    -------
    loads
        ``GetLoadResult`` objects or ``None`` for each server.
    """
    # Asynchronously get the current load information for each server
    get_load_coros = [get_load_async(host, port, timeout) for host, port in hosts_and_ports]
    # Complete the asynchronous load queries
    loads = await asyncio.gather(*get_load_coros, return_exceptions=True)
    # Replace exceptions by None
    return [(l if isinstance(l, GetLoadResult) else None) for l in loads]


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
        # For randomization of order and delays we need thread-safe random numbers!
        rng = np.random.default_rng(seed=random.randint(0, 100_000))

        # First shuffle the list. This achieves two things:
        # 1. Our queries are send in random order
        # 2. If two servers have identical load, a random one will be selected.
        hosts_and_ports = rng.permutation(hosts_and_ports)

        # Now wait a random interval to de-synchronize from parallel processes.
        await asyncio.sleep(rng.uniform(0.2, 2))

        # Asynchronously retrieve load information from all servers
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


async def _connect_evaluate_async(
    input: InputArrays,
    cid: str,
    hosts_and_ports: Sequence[Tuple[str, int]],
    use_stream: bool,
) -> OutputArrays:
    """Connect (or re-use connection) and evaluate.

    Parameters
    ----------
    input
        Input message for the evaluation call.
    cid
        Thread- and process-unique identifier of the client object.
    hosts_and_ports
        List of (host, port) tuples.
    use_stream
        If `True`, the evaluation uses a bidirectional stream
        instead of opening & closing a stream just for this call.

    Returns
    -------
    output
        Result message of the evaluation call.
    """
    if not cid in _privates:
        _log.debug("Connecting client %s", cid)
        if len(hosts_and_ports) == 1:
            host, port = hosts_and_ports[0]
            connect_coro = ClientPrivates.connect(host, port)
        else:
            connect_coro = ClientPrivates.connect_balanced(hosts_and_ports)
        pc = await connect_coro
        _privates[cid] = pc
        _log.info("Client %s connected to %s:%s", cid, pc.channel._host, pc.channel._port)
    priv = _privates[cid]

    # Make the asynchronous calls to the remote server
    if use_stream:
        eval_coro = _streamed_evaluate(input, priv.stream)
    else:
        eval_coro = priv.client.evaluate(input)
    output = await eval_coro
    if output.uuid != input.uuid:
        raise Exception("Response does not correspond to the request.")
    return output


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
            loop = get_useful_event_loop()
            loop.run_until_complete(priv.stream.end())
            priv.channel.close()
            # Remove from the dict, otherwise it won't be garbage-collected
            del _privates[_id]
        return

    def __call__(self, *inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Alias for ``.evaluate(*inputs)``."""
        return self.evaluate(*inputs)

    def evaluate(self, *inputs: Sequence[np.ndarray], **kwargs) -> Sequence[np.ndarray]:
        loop = get_useful_event_loop()
        eval_coro = self.evaluate_async(*inputs, **kwargs)
        return loop.run_until_complete(eval_coro)

    async def evaluate_async(
        self, *inputs: Sequence[np.ndarray], use_stream: bool = True, retries: int = 2
    ) -> Sequence[np.ndarray]:
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
        if retries < 0:
            raise ValueError("Number of retries must be >= 0.")

        # Encode inputs
        input = InputArrays(
            items=[ndarray_from_numpy(np.asarray(i)) for i in inputs], uuid=str(uuid.uuid4())
        )

        # Get a unique ID of this client instance.
        # This is used by `_connect_evaluate_async` to cache the gRPC connection objects.
        cid = thread_pid_id(self)
        hap = self._hosts_and_ports or [(self._host, self._port)]

        output = None
        for r in range(retries + 1):
            try:
                output = await _connect_evaluate_async(input, cid, hap, use_stream)
                break
            except grpclib.exceptions.StreamTerminatedError:
                if cid in _privates:
                    cp = _privates.pop(cid)
                    _log.warning("Lost connection to %s:%s.", cp.channel._host, cp.channel._port)
                    cp.channel.close()

        # If no servers are available, already the connection setup fails.
        assert output is not None

        # Decode outputs
        outputs = [ndarray_to_numpy(o) for o in output.items]
        return outputs
