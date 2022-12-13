import multiprocessing
import platform
import time
from typing import Sequence
from unittest import mock

import grpclib
import numpy as np
import pytest

from pytensor_federated import service, signatures
from pytensor_federated.npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from pytensor_federated.rpc import GetLoadResult, InputArrays, OutputArrays
from pytensor_federated.utils import get_useful_event_loop


def test_compute_function():
    def compute_fun(a, b):
        cost = np.sum(a + b)
        grads = [
            np.prod(a + b),
            np.prod(a - b),
        ]
        return cost, *grads

    # Assembly the input protobuf
    a = np.array([1, 2])
    b = np.array([3, 4.5])
    input = InputArrays(
        items=[
            ndarray_from_numpy(a),
            ndarray_from_numpy(b),
        ]
    )
    output = service._run_compute_func(input, compute_fun)

    # Assert on the returned result protobuf
    assert isinstance(output, OutputArrays)
    np.testing.assert_array_equal(ndarray_to_numpy(output.items[0]), 10.5)
    np.testing.assert_array_equal(ndarray_to_numpy(output.items[1]), 4 * 6.5)
    np.testing.assert_array_equal(ndarray_to_numpy(output.items[2]), -2 * -2.5)
    pass


def product_func(*inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Calculates the product of NumPy arrays"""
    return (np.prod(inputs),)


def run_service(port: int, compute_func: signatures.ComputeFunc, n_clients: int = 0):
    """Serve a compute_func in a local gRPC server"""

    async def run_server():
        a2a_service = service.ArraysToArraysService(compute_func)
        # Override the number of clients for testing purposes
        a2a_service._n_clients = n_clients

        server = grpclib.server.Server([a2a_service])
        await server.start("127.0.0.1", port)
        await server.wait_closed()

    loop = get_useful_event_loop()
    loop.run_until_complete(run_server())
    return


def run_product_queries(client: service.ArraysToArraysServiceClient, n=100):
    """Runs n random product calculations using the client for the `product_func`."""
    try:
        for _ in range(n):
            a, b = np.random.randint(0, 50, size=2)
            prod = client.evaluate(a, b)
            assert prod == a * b
    except:
        return False
    return True


class ProductTester:
    def __init__(self, client: service.ArraysToArraysServiceClient) -> None:
        self.client = client
        pass

    def run(self, n: int):
        return run_product_queries(self.client, n)


@mock.patch("psutil.getloadavg", return_value=[0.1, 0.2, 0.3])
@mock.patch("psutil.cpu_count", return_value=3)
def test_determine_load(mock_getloadavg, mock_cpu_count):
    a2a_service = service.ArraysToArraysService(product_func)

    # The load mus tbe determined once at startup
    # to make psutil.getloadavg start monitoring.
    mock_getloadavg.assert_called_once_with()
    mock_cpu_count.assert_called_once_with()

    # Number of clients and CPU load are set to known values
    # through the mocks, and the next line.
    a2a_service._n_clients = 3
    load = a2a_service.determine_load()
    assert load.n_clients == 3
    assert load.percent_cpu == 0.1 / 3 * 100
    # RAM load is not mocked
    assert 0 < load.percent_ram < 100
    pass


def test_get_loads_async():
    server_processes = [
        multiprocessing.Process(target=run_service, args=[9500 + p, product_func, nc])
        for p, nc in enumerate([3, 4, 2])
    ]
    try:
        for sp in server_processes:
            sp.start()
        time.sleep(5)

        loads_task = service.get_loads_async(
            [
                ("127.0.0.1", 9499),  # this one is offline
                ("127.0.0.1", 9500),
                ("127.0.0.1", 9501),
                ("127.0.0.1", 9502),
            ]
        )
        loop = get_useful_event_loop()
        loads = loop.run_until_complete(loads_task)
        assert isinstance(loads, list)
        assert loads[0] is None
        for l in loads[1:]:
            assert isinstance(l, GetLoadResult)
        assert loads[1].n_clients == 3
        assert loads[2].n_clients == 4
        assert loads[3].n_clients == 2
    finally:
        # Always stop the server again
        for sp in server_processes:
            sp.terminate()
            sp.join()
    pass


def test_client_loadbalancing():
    server_processes = [
        multiprocessing.Process(target=run_service, args=[9500 + p, product_func, nc])
        for p, nc in enumerate([3, 4, 2])
    ]
    try:
        for sp in server_processes:
            sp.start()
        time.sleep(5)

        client = service.ArraysToArraysServiceClient(
            hosts_and_ports=[
                ("127.0.0.1", 9499),  # this one is offline
                ("127.0.0.1", 9500),
                ("127.0.0.1", 9501),
                ("127.0.0.1", 9502),  # this one reports the fewest n_clients
            ]
        )
        # Use the client to trigger connection setup
        result = client.evaluate(np.array(2), np.array(3))
        assert isinstance(result, list)
        assert isinstance(result[0], np.ndarray)
        assert result[0] == 6
        # Assert which server it connected to
        cid = service.thread_pid_id(client)
        assert cid in service._privates
        privs = service._privates[cid]
        assert privs.channel._port == "9502"
    finally:
        # Always stop the server again
        for sp in server_processes:
            sp.terminate()
            sp.join()
    pass


@pytest.mark.parametrize("eval_on_main", [False, True])
@pytest.mark.parametrize("mp_start_method", ["spawn", "fork"])
def test_client_multiprocessing(eval_on_main, mp_start_method):
    """
    This test runs a server in a child process, and starts
    a pool of child processes to query the server.
    In the end, the server process is stopped regardless.
    """
    # on Windows we cannot fork
    if platform.system() == "Windows" and mp_start_method == "fork":
        return

    ctx = multiprocessing.get_context(mp_start_method)

    p_server = ctx.Process(
        target=run_service,
        args=(
            9321,
            product_func,
        ),
    )
    try:
        p_server.start()
        time.sleep(5)

        client = service.ArraysToArraysServiceClient("127.0.0.1", 9321)
        tester = ProductTester(client)
        if eval_on_main:
            tester.run(n=2)

        # Passing clients as args
        with ctx.Pool(processes=3) as pool:
            results = pool.map(run_product_queries, [client] * 5)
            assert all(results)

        # Passing client through the callable
        with ctx.Pool(processes=3) as pool:
            results = pool.map(tester.run, [25] * 5)
            assert all(results)

    finally:
        # Always stop the server again
        p_server.terminate()
        p_server.join()
    pass


def test_client_argchecking():
    client = service.ArraysToArraysServiceClient("localhost", 9999)
    with pytest.raises(ValueError, match="must be >= 0"):
        client.evaluate(np.array(0), np.array(1), retries=-1)
    pass


def test_client_failover():
    p0 = multiprocessing.Process(target=run_service, args=[9400, product_func, 5])
    p1 = multiprocessing.Process(target=run_service, args=[9401, product_func, 2])
    try:
        p0.start()
        p1.start()
        time.sleep(5)

        client = service.ArraysToArraysServiceClient(
            hosts_and_ports=[
                ("127.0.0.1", 9400),
                ("127.0.0.1", 9401),  # fewest n_clients
            ]
        )
        cid = service.thread_pid_id(client)

        # First evaluation creates the connection to the least-busy server
        assert cid not in service._privates
        client.evaluate(np.array(2), np.array(3))[0] == 6
        assert cid in service._privates
        channel1 = service._privates[cid].channel
        assert channel1._port == "9401"

        # Killing the server is not immediately noticed by the client
        p1.terminate()
        p1.join()
        assert service._privates[cid].channel._port == "9401"

        # Evaluating with one retry attempt fails over to the other server
        client.evaluate(np.array(2), np.array(4), retries=1)[0] == 8
        # Confirm that the previous channel it was properly closed
        assert channel1._protocol is None
        # The new connection should have failed over to the third server
        assert service._privates[cid].channel._port == "9400"

        # Now kill that one too
        p0.terminate()
        p0.join()

        # No servers left
        with pytest.raises(TimeoutError, match="None of 2 servers responded"):
            client.evaluate(np.array(2), np.array(4))

    finally:
        # Always stop the server again
        p0.terminate()
        p1.terminate()
        p0.join()
        p1.join()
    pass
