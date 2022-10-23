import asyncio
import time

import aesara
import aesara.tensor as at
import pytest
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op

from aesara_federated import op_async


class _AsyncDelay(op_async.AsyncOp):
    def make_node(self, delay: Variable) -> Apply:
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
        f = aesara.function([d], [out])
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
        f = aesara.function([d1, d2], [dsum])
        t_start = time.perf_counter()
        delay_sum = f(0.5, 0.2)[0]
        t_took = time.perf_counter() - t_start
        assert float(delay_sum) == 0.7
        assert 0.5 < t_took < delay_sum
        pass
