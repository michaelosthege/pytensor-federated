import asyncio
import time
from typing import Sequence, Tuple

import numpy
import pytensor
import pytensor.tensor as at
import pytest
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op

from pytensor_federated import op_async


class _AsyncDelay(op_async.AsyncOp):
    def make_node(self, delay: Variable) -> Apply:
        delay = at.as_tensor(delay)
        return Apply(
            op=self,
            inputs=[delay],
            outputs=[at.scalar()],
        )

    async def perform_async(self, node, inputs, output_storage, params=None) -> None:
        delay = inputs[0]
        # The asyncio.sleep function is not accurate in comparison to the perf_counter.
        # Therefore, this implementation uses the perf_counter such that the outer
        # test code does not get confused by slightly too short delays.
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < delay:
            await asyncio.sleep(0.01)
        output_storage[0][0] = delay


class TestAsyncOp:
    def test_perform(self):
        delay_op = _AsyncDelay()
        assert isinstance(delay_op, Op)

        d = at.scalar()
        out = delay_op(d)
        # Compile a function to exclude compile time from delay measurement
        f = pytensor.function([d], [out])
        ts = time.perf_counter()
        f(0.5)
        assert 0.5 < time.perf_counter() - ts < 0.6
        pass


class TestAsyncFromFunctionOp:
    def test_performs(self):
        async def _fn(x):
            await asyncio.sleep(x)
            return x

        affo = op_async.AsyncFromFunctionOp(
            fn=_fn,
            itypes=[at.dscalar],
            otypes=[at.dscalar],
        )
        assert isinstance(affo, op_async.AsyncOp)
        assert isinstance(affo, pytensor.compile.ops.FromFunctionOp)

        d = at.scalar()
        out = affo(d)
        # Compile a function to exclude compile time from delay measurement
        f = pytensor.function([d], [out])
        ts = time.perf_counter()
        f(0.5)
        assert 0.5 < time.perf_counter() - ts < 0.6
        pass


class TestParallelAsyncOp:
    def test_perform(self):
        # Create two nodes that are the result of separate applies
        d1 = at.scalar()
        d2 = at.scalar()
        dop = _AsyncDelay()
        o1 = dop(d1)
        o2 = dop(d2)

        # Assert that the constructor checks the input owner types
        with pytest.raises(ValueError, match="apply node 2 is not"):
            op_async.ParallelAsyncOp([o1.owner, o2.owner, (o1 + o2).owner])

        # Create a parallelized Op for these applies
        pop = op_async.ParallelAsyncOp(applies=[o1.owner, o2.owner])

        # Assert that make_node checks the number of inputs
        with pytest.raises(ValueError, match="expected 2 for 2"):
            pop(d1, d2, 3)

        outs = pop(d1, d2)
        dsum = outs[0] + outs[1]

        # Evaluating the delays in parallel is faster than the sum of delays.
        # We do this with a compiled function to exclude compile time from delay measurement.
        f = pytensor.function([d1, d2], [dsum])
        t_start = time.perf_counter()
        delay_sum = f(0.5, 0.2)[0]
        t_took = time.perf_counter() - t_start
        assert float(delay_sum) == 0.7
        assert 0.5 < t_took < delay_sum
        pass


def test_find_parallelizable_applies():
    a, b, c = at.scalars("abc")
    x = at.sum([a + 1, b + 2, c + 3]) + 4
    fg = FunctionGraph([a, b, c], [x])

    found = op_async.find_parallelizable_applies(fg, at.elemwise.Elemwise)
    assert isinstance(found, list)
    # The three +1 operations are independent
    assert len(found) == 3
    assert all(isinstance(a.op, at.elemwise.Elemwise) for a in found)
    pass


def test_parallelize_async_applies():
    delay = _AsyncDelay()
    a, b = at.scalars("ab")
    c = delay(delay(a) + delay(b))

    fg = FunctionGraph([a, b], [c])

    # check the structure of the original graph
    a3 = fg.outputs[0].owner
    d1, d2 = a3.inputs[0].owner.inputs
    a1 = d1.owner
    a2 = d2.owner
    assert isinstance(a1.op, _AsyncDelay)
    assert isinstance(a2.op, _AsyncDelay)
    assert isinstance(a3.op, _AsyncDelay)
    assert a1 is not a2

    # fuse the applies producing d1 and d2
    op_async.parallelize_async_applies(fg, [a1, a2])
    assert any(isinstance(a.op, op_async.ParallelAsyncOp) for a in fg.apply_nodes)

    # The inputs to the sum were replaced by new variables
    n1, n2 = fg.outputs[0].owner.inputs[0].owner.inputs
    assert n1 is not d1
    # These new nodes were produced by a ParallelAsyncOp
    assert n1.owner is n2.owner
    assert isinstance(n1.owner.op, op_async.ParallelAsyncOp)
    assert set(n1.owner.op.applies) == {a1, a2}
    pass


def _measure_fg(fg: FunctionGraph, *inputs) -> Tuple[Sequence[numpy.ndarray], float]:
    """Measure the runtime of a function compiled from `fg`."""
    f = pytensor.function(
        fg.inputs,
        fg.outputs,
        mode=pytensor.compile.FAST_COMPILE,  # skip the async fusion
    )
    t0 = time.perf_counter()
    outputs = f(*inputs)
    dt = time.perf_counter() - t0
    return outputs, dt


def test_parallelize_all_async_applies():
    delay = _AsyncDelay()
    a, b = at.scalars("ab")

    # Two parallel delays
    ab = delay(a) + delay(b)
    # Another two parallel delays that depend on the previous layer
    total = delay(ab) + delay(ab + 1)
    # Total delays
    # sequential: a + b + (a + b) + (a + b + 1)
    # parallel  : max(ab) + (a + b + 1)

    fg = FunctionGraph([a, b], [total])
    assert sum(isinstance(a.op, _AsyncDelay) for a in fg.apply_nodes) == 4

    # Time it sequentially
    _, act = _measure_fg(fg, 0.3, 0.7)
    exp = 3 * 0.3 + 3 * 0.7 + 1  # 4 seconds
    assert exp < act < exp + 0.2

    # Now optimize
    op_async.parallelize_all_async_applies(fg)
    assert not any(isinstance(a.op, _AsyncDelay) for a in fg.apply_nodes)
    assert sum(isinstance(a.op, op_async.ParallelAsyncOp) for a in fg.apply_nodes) == 2

    # Time it in parallel
    _, act = _measure_fg(fg, 0.3, 0.7)
    exp = 0.7 + 0.3 + 0.7 + 1  # 2.7 seconds
    assert exp < act < exp + 0.2
    pass


def test_fuse_asyncs_by_default():
    delay = _AsyncDelay()
    a, b = at.scalars("ab")
    c = delay(a) + delay(b)
    f = pytensor.function([a, b], [c])
    t0 = time.perf_counter()
    f(0.25, 0.25)
    assert time.perf_counter() - t0 < 0.3
    pass
