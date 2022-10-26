import asyncio
import logging
from typing import Any, Sequence

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op, OutputStorageType, ParamsInputType

from .utils import get_useful_event_loop


class AsyncOp(Op):
    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        loop = get_useful_event_loop()
        coro = self.perform_async(node, inputs, output_storage, params)
        loop.run_until_complete(coro)
        return

    async def perform_async(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        raise NotImplementedError()


class ParallelAsyncOp(AsyncOp):
    """An op that parallelizes the `perform_async` methods of multiple `AsyncOp` apply nodes."""

    def __init__(self, applies: Sequence[Apply]) -> None:
        for a, apply in enumerate(applies):
            if not isinstance(apply.op, AsyncOp):
                raise ValueError(
                    f"The owner of apply node {a} is not an `AsyncOp`. "
                    "All apply nodes given to given to `ParallelAsyncOp` must be owned by an `AsyncOp`."
                )
        self.applies = applies
        super().__init__()

    def make_node(self, *inputs: Variable) -> Apply:
        # Check number of inputs
        nin_exp = sum(a.nin for a in self.applies)
        nin_act = len(inputs)
        if nin_act != nin_exp:
            raise ValueError(
                f"Unexpected number of inputs to `ParallelAsyncOp` {self}. "
                f"Got {nin_act} inputs but expected {nin_exp} for {len(self.applies)} apply nodes."
            )

        # Create new unowned output variables
        outputs = []
        for app in self.applies:
            for out in app.outputs:
                outputs.append(out.type())

        # Create a new apply node that takes does the job of all child applies at once.
        return Apply(
            op=self,
            inputs=inputs,
            outputs=outputs,
        )

    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        # Create coroutines the performing the taks of each child node
        coros = []
        ifrom = 0
        ofrom = 0
        for apply in self.applies:
            ito = ifrom + apply.nin
            oto = ofrom + apply.nout
            coros.append(
                apply.op.perform_async(
                    apply, inputs[ifrom:ito], output_storage[ofrom:oto], params=params
                )
            )
            ifrom = ito
            ofrom = oto

        # Wait for completion of all sub-performs
        loop = get_useful_event_loop()
        futures = [asyncio.ensure_future(c, loop=loop) for c in coros]
        pool = asyncio.gather(*futures, return_exceptions=True)
        loop.run_until_complete(pool)
        # Output storage was modified inplace by the child operations.
        return
