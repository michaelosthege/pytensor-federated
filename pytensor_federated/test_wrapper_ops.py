import asyncio
import logging
import multiprocessing
import sys
import time

import arviz
import grpclib
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as at
import pytest
import scipy
from pytensor.compile.ops import FromFunctionOp
from pytensor.graph.basic import Apply, Variable

from pytensor_federated import common, op_async, service, wrapper_ops
from pytensor_federated.utils import get_useful_event_loop


class _MockLogpGradOpClient:
    def __init__(self, fn) -> None:
        self._fn = fn
        super().__init__()

    def evaluate(self, *inputs):
        return self._fn(*inputs)

    def __call__(self, *inputs):
        return self.evaluate(*inputs)


def dummy_quadratic_model(a, b):
    """Dummy model with sum of squared residuals and manual gradients."""
    rng = np.random.RandomState(42)
    x = np.array([1, 2, 3])
    y = rng.normal(2 * x**2 + 0.5, scale=0.1)
    pred = a * x**2 + b
    cost = np.asarray(np.sum((pred - y) ** 2))
    grad = [
        np.asarray(np.sum(2 * x**2 * (a * x**2 + b - y))),
        np.asarray(np.sum(2 * (a * x**2 + b - y))),
    ]
    return cost, grad


def linear_and_quadratic_compute_func(a, b, x):
    """Calculates 1st and 2nd order polynomial at the same time."""
    linear = a + b * x
    quadratic = a + b * x**2
    return linear, quadratic


def blackbox_linear_model(intercept, slope):
    """Blackbox likelihood of a linear model.

    There are 15 observations and the groundtruth is a=2, b=0.5.
    """
    rng = np.random.RandomState(42)
    x = np.linspace(-3, 3, 15, dtype=float)
    y = rng.normal(2 * x + 0.5, scale=0.1)
    pred = slope * x + intercept
    L = scipy.stats.norm.logpdf(loc=pred, scale=0.1, x=y)
    return np.asarray(L.sum())


def run_blackbox_linear_model_service(port: int):
    async def run_server():
        a2a_service = service.ArraysToArraysService(common.wrap_logp_func(blackbox_linear_model))
        server = grpclib.server.Server([a2a_service])
        await server.start("127.0.0.1", port)
        await server.wait_closed()

    loop = get_useful_event_loop()
    loop.run_until_complete(run_server())
    return


def run_blackbox_linear_model_mcmc(port: int, cores: int, use_async: bool):
    client = common.LogpServiceClient("127.0.0.1", port)

    # Choose the callable and corresponding wrapper Op
    if use_async:
        fn = client.evaluate_async
        op_cls = wrapper_ops.AsyncLogpOp
    else:
        fn = client
        op_cls = wrapper_ops.LogpOp

    # Do the check on the main process to keep tracebacks readable.
    result = client(0.4, 1.2)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, -1511.41423640139)

    # Create the Op, build and sample the PyMC model
    blackbox_L = op_cls(fn)
    with pm.Model():
        # This runs with Metropolis which is inefficient.
        # Let's make it easy...
        intercept = at.constant(0.5)
        slope = pm.Normal("slope", sigma=2)
        L = blackbox_L(intercept, slope)
        pm.Potential("L", L)
        idata = pm.sample(
            tune=200,
            chains=3,
            cores=cores,
            step=pm.Metropolis(),
            compute_convergence_checks=False,
            random_seed=1234,
        )
    # Check the posterior medians against the ground truth.
    # Print the summary so failure logs are more informative.
    print(arviz.summary(idata))
    pst = idata.posterior.stack(sample=("chain", "draw"))
    np.testing.assert_allclose(np.median(pst.slope), 2, atol=0.1)
    return


class TestArraysToArraysOp:
    def test_basics(self):
        fn = linear_and_quadratic_compute_func
        itypes = [at.dscalar, at.dscalar, at.dvector]
        otypes = [at.dvector, at.dvector]

        assert issubclass(wrapper_ops.ArraysToArraysOp, FromFunctionOp)
        ataop = wrapper_ops.ArraysToArraysOp(fn, itypes, otypes)
        assert isinstance(ataop, FromFunctionOp)

        # Passing dtypes is needed because of OS-specific float32/64 defaults.
        a = np.array(2, dtype="float64")
        b = np.array(3, dtype="float64")
        x = np.arange(4, dtype="float64")
        y1, y2 = ataop(a, b, x)
        expected = fn(a, b, x)
        np.testing.assert_array_equal(y1.eval(), expected[0])
        np.testing.assert_array_equal(y2.eval(), expected[1])
        pass


class TestAsyncArraysToArraysOp:
    def test_basics(self):
        async def fn(*args):
            await asyncio.sleep(0.001)
            return linear_and_quadratic_compute_func(*args)

        itypes = [at.dscalar, at.dscalar, at.dvector]
        otypes = [at.dvector, at.dvector]

        assert issubclass(wrapper_ops.AsyncArraysToArraysOp, op_async.AsyncOp)
        assert issubclass(wrapper_ops.AsyncArraysToArraysOp, FromFunctionOp)
        ataop = wrapper_ops.AsyncArraysToArraysOp(fn, itypes, otypes)
        assert isinstance(ataop, FromFunctionOp)

        # Passing dtypes is needed because of OS-specific float32/64 defaults.
        a = np.array(2, dtype="float64")
        b = np.array(3, dtype="float64")
        x = np.arange(4, dtype="float64")
        y1, y2 = ataop(a, b, x)
        expected = linear_and_quadratic_compute_func(a, b, x)
        np.testing.assert_array_equal(y1.eval(), expected[0])
        np.testing.assert_array_equal(y2.eval(), expected[1])
        pass


class TestLogpGradOp:
    def test_init(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = wrapper_ops.LogpGradOp(client)
        assert flop._logp_grad_func is client
        pass

    def test_make_node(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = wrapper_ops.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        apply = flop.make_node(a, b)
        assert isinstance(apply, Apply)
        assert len(apply.inputs) == 2
        assert len(apply.outputs) == 3

        # Test with non-Variable inputs (issue #24)
        apply = flop.make_node(1, 2)
        assert isinstance(apply, Apply)
        assert all(isinstance(i, Variable) for i in apply.inputs)
        pass

    def test_perform(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = wrapper_ops.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        apply = flop.make_node(a, b)
        inputs = [np.array(1.2), np.array(1.5)]
        output_storage = [[None], [None], [None]]
        flop.perform(apply, inputs, output_storage)
        logp, (da, db) = dummy_quadratic_model(*inputs)
        np.testing.assert_array_equal(output_storage[0][0], logp)
        np.testing.assert_array_equal(output_storage[1][0], da)
        np.testing.assert_array_equal(output_storage[2][0], db)
        pass

    def test_forward(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = wrapper_ops.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        logp, da, db = flop(a, b)
        assert isinstance(logp, Variable)
        assert isinstance(da, Variable)
        assert isinstance(db, Variable)
        inputvals = {
            a: 1.2,
            b: 1.7,
        }
        exlogp, (exda, exdb) = dummy_quadratic_model(1.2, 1.7)
        assert logp.eval(inputvals) == exlogp
        assert da.eval(inputvals) == exda
        assert db.eval(inputvals) == exdb
        pass

    def test_grad(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = wrapper_ops.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        logp, *_ = flop(a, b)
        ga, gb = at.grad(logp, [a, b])
        fn = pytensor.function(inputs=[a, b], outputs=[logp, ga, gb])
        exlogp, (exda, exdb) = dummy_quadratic_model(1.4, 0.5)
        actual = fn(1.4, 0.5)
        np.testing.assert_array_equal(actual[0], exlogp)
        np.testing.assert_array_equal(actual[1], exda)
        np.testing.assert_array_equal(actual[2], exdb)
        pass


class TestAsyncLogpGradOp:
    def test_perform_async(self):
        async def fn(*args):
            await asyncio.sleep(0.001)
            return dummy_quadratic_model(*args)

        flop = wrapper_ops.AsyncLogpGradOp(fn)
        assert isinstance(flop, op_async.AsyncOp)
        assert isinstance(flop, wrapper_ops.LogpGradOp)
        a = at.scalar()
        b = at.scalar()
        apply = flop.make_node(a, b)
        inputs = [np.array(1.2), np.array(1.5)]
        output_storage = [[None], [None], [None]]
        flop.perform(apply, inputs, output_storage)
        logp, (da, db) = dummy_quadratic_model(*inputs)
        np.testing.assert_array_equal(output_storage[0][0], logp)
        np.testing.assert_array_equal(output_storage[1][0], da)
        np.testing.assert_array_equal(output_storage[2][0], db)
        pass


class TestLogpOp:
    @pytest.mark.parametrize("use_async", [False, True])
    def test_make_node(self, use_async):
        def fn(a, b):
            return a + b

        async def fn_async(a, b):
            await asyncio.sleep(0.001)
            return a + b

        if use_async:
            lop = wrapper_ops.LogpOp(fn)
        else:
            lop = wrapper_ops.AsyncLogpOp(fn_async)

        a = at.scalar()
        b = at.scalar()
        apply = lop.make_node(a, b)
        assert isinstance(apply, Apply)
        assert len(apply.inputs) == 2
        assert len(apply.outputs) == 1

        # Test with non-Variable inputs (issue #24)
        apply = lop.make_node(1, 2)
        assert isinstance(apply, Apply)
        assert all(isinstance(i, Variable) for i in apply.inputs)
        assert lop(1, 2).eval() == 3
        pass

    def test_pymc_sampling_sequential(self):
        # Launch a blackbox loglikelihood on a child process.
        port = 9130
        p_server = multiprocessing.Process(target=run_blackbox_linear_model_service, args=(port,))
        try:
            p_server.start()
            time.sleep(10)
            run_blackbox_linear_model_mcmc(port, cores=1, use_async=False)
        finally:
            # Always stop the server again
            p_server.terminate()
            p_server.join()
        pass

    def test_pymc_sampling_parallel(self):
        # Launch a blackbox loglikelihood on a child process.
        port = 9130
        p_server = multiprocessing.Process(target=run_blackbox_linear_model_service, args=(port,))
        try:
            p_server.start()
            time.sleep(10)
            run_blackbox_linear_model_mcmc(port, cores=4, use_async=True)
        finally:
            # Always stop the server again
            p_server.terminate()
            p_server.join()
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        raise Exception("Pass 'service' or 'mcmc' as the first argument.")
    if sys.argv[1] == "service":
        run_blackbox_linear_model_service(port=9130)
    elif sys.argv[1] == "mcmc":
        run_blackbox_linear_model_mcmc(port=9130, cores=4, use_async=True)
