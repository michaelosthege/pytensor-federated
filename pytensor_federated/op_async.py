import asyncio
from typing import Any, Callable, List, Optional, Sequence

import pytensor.tensor as at
from pytensor.compile import optdb
from pytensor.compile.ops import FromFunctionOp
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Apply, Variable, apply_depends_on
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.op import Op, OutputStorageType
from pytensor.graph.rewriting.basic import GraphRewriter

from .utils import get_useful_event_loop


class AsyncOp(Op):
    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        loop = get_useful_event_loop()
        coro = self.perform_async(node, inputs, output_storage)
        loop.run_until_complete(coro)
        return

    async def perform_async(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        raise NotImplementedError()


class AsyncFromFunctionOp(AsyncOp, FromFunctionOp):
    """Async version of the ``pytensor.compile.ops.FromFunctionOp``.

    Note that ``AsyncOp.perform`` overrides ``FromFunctionOp.perform`` by MRO.
    """

    def __init__(
        self,
        fn: Callable,
        itypes: Sequence[at.TensorType],
        otypes: Sequence[at.TensorType],
        infer_shape: Optional[Callable] = None,
    ):
        self.__async_fn = fn
        super().__init__(fn, itypes, otypes, infer_shape)

    async def perform_async(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        outs = await self.__async_fn(*inputs)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)
        assert len(outs) == len(output_storage)
        for i in range(len(outs)):
            output_storage[i][0] = outs[i]
        return


class ParallelAsyncOp(AsyncOp):
    """An op that parallelizes the `perform_async` methods of multiple `AsyncOp` apply nodes."""

    def __init__(self, applies: Sequence[Apply]) -> None:
        # Freeze the order of items
        applies = tuple(applies)
        # Confirm that all were the result of an AsyncOp
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
    ) -> None:
        # Create coroutines the performing the taks of each child node
        coros = []
        ifrom = 0
        ofrom = 0
        for apply in self.applies:
            ito = ifrom + apply.nin
            oto = ofrom + apply.nout
            coros.append(
                apply.op.perform_async(apply, inputs[ifrom:ito], output_storage[ofrom:oto])
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


def find_parallelizable_applies(fg: FunctionGraph, op_cls: type) -> List[Apply]:
    """Searches for independent ``Apply`` nodes of type ``op_cls``.

    Parameters
    ----------
    fg
        A ``FunctionGraph`` to search.
    op_cls
        A subtype of ``Op`` to search for.

    Returns
    -------
    applies
        A possibly empty sequence of ``Apply`` nodes that could compute in parallel.
    """
    applies = []
    for apply in fg.toposort():
        if not isinstance(apply.op, op_cls):
            continue
        # Does this node depend on any one of the already-collected?
        if not any(apply_depends_on(apply, a) for a in applies):
            applies.append(apply)
            continue
        elif len(applies) == 1:
            # Only one node collected already, but the current depends on it.
            # Start over starting from the current one.
            applies = [apply]
        else:
            # Multiple nodes upstream of the current one can be parallelized.
            break
    if len(applies) > 1:
        return applies
    return []


def parallelize_async_applies(fg: FunctionGraph, applies: Sequence[Apply]) -> None:
    """Combines multiple ``Apply`` nodes into one produced by a ``ParallelAsyncOp``.

    Parameters
    ----------
    fg
        The ``FunctionGraph`` to edit.
    applies
        A sequence of parallelizable ``Apply`` nodes
        that resulted from ``AsyncOp``s.
    """
    # Concatenate inputs and outputs
    inputs = []
    old_outputs = []
    for apply in applies:
        inputs.extend(apply.inputs)
        old_outputs.extend(apply.outputs)
    # Push the original inputs through the parallel Op to obtain new outputs.
    pop = ParallelAsyncOp(applies=applies)
    new_outputs = pop(*inputs)
    # Replace old output variables with the new ones to substitute the applies.
    # If the graph has the `ReplaceValidate`, we prefer to use
    # `replace_all_validate` which runs some safety checks.
    replace_all = getattr(fg, "replace_all_validate", fg.replace_all)
    replace_all(list(zip(old_outputs, new_outputs)))
    return


def parallelize_all_async_applies(fg: FunctionGraph):
    """Recursively fuse parallelizable ``AsyncOp`` applications inplace.

    This optimizes the graph for faster execution
    by running ``perform_async`` in parallel where possible.

    Parameters
    ----------
    fg
        A ``FunctionGraph`` to rewrite inplace.
    """
    applies = find_parallelizable_applies(fg, AsyncOp)
    while applies:
        parallelize_async_applies(fg, applies)
        applies = find_parallelizable_applies(fg, AsyncOp)
    return


class AsyncFusionOptimizer(GraphRewriter):
    """Optimizer that parallelizes ``AsyncOp.perform_async`` calls."""

    def add_requirements(self, fgraph: FunctionGraph):
        """Enable node replacements with safety checks."""
        fgraph.attach_feature(ReplaceValidate())

    def apply(self, fgraph: FunctionGraph):
        parallelize_all_async_applies(fgraph)


# Register the parallelization as a graph optimization to run in `FAST_RUN` compile modes.
if not "fuse_asyncs" in optdb:
    optdb.register(
        "fuse_asyncs",
        AsyncFusionOptimizer(),
        "fast_run",
        position=90,
    )
