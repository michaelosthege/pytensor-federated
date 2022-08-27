import asyncio
import multiprocessing
import platform
import time
from typing import Sequence

import grpclib
import numpy as np
import pytest

from aesara_federated import service, signatures
from aesara_federated.npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from aesara_federated.rpc import InputArrays, OutputArrays


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


def run_service(port: int, compute_func: signatures.ComputeFunc):
    """Serve a compute_func in a local gRPC server"""

    async def run_server():
        a2a_service = service.ArraysToArraysService(compute_func)
        assert a2a_service._n_clients == 0
        server = grpclib.server.Server([a2a_service])
        await server.start("127.0.0.1", port)
        await server.wait_closed()

    loop = asyncio.get_event_loop()
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
