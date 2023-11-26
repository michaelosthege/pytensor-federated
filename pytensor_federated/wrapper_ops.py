from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pytensor
import pytensor.tensor as at
from pytensor.compile.ops import FromFunctionOp
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op, OutputStorageType

from .op_async import AsyncFromFunctionOp, AsyncOp
from .signatures import ComputeFunc, LogpFunc, LogpGradFunc


class ArraysToArraysOp(FromFunctionOp):
    """Alias for the `pytensor.compile.ops.FromFunctionOp`.

    This alias exists for more convenient imports,
    more informative type hints,
    and for continuitiy with other Ops from this package.
    """

    def __init__(
        self,
        compute_func: ComputeFunc,
        itypes: Sequence[at.TensorType],
        otypes: Sequence[at.TensorType],
        infer_shape: Optional[Callable] = None,
    ):
        super().__init__(compute_func, itypes, otypes, infer_shape)

    def make_node(self, *inputs: Variable) -> Apply:
        input_tensors = list(map(at.as_tensor, inputs))
        return super().make_node(*input_tensors)


class AsyncArraysToArraysOp(AsyncFromFunctionOp):
    """Async equivalent to ``ArraysToArraysOp``."""

    def make_node(self, *inputs: Variable) -> Apply:
        input_tensors = list(map(at.as_tensor, inputs))
        return super().make_node(*input_tensors)


class LogpOp(Op):
    """An Op that wraps a callable returning a log-potential."""

    _props = ("_logp_func",)

    def __init__(self, logp_func: LogpFunc) -> None:
        self._logp_func = logp_func
        super().__init__()

    def make_node(self, *inputs: Union[Variable, int, float, np.ndarray]) -> Apply:
        logp = at.scalar()
        return Apply(
            op=self,
            inputs=list(map(at.as_tensor, inputs)),
            outputs=[logp],
        )

    def perform(
        self,
        node: Apply,
        inputs: Sequence[np.ndarray],
        output_storage: OutputStorageType,
    ) -> None:
        logp = self._logp_func(*inputs)
        output_storage[0][0] = logp
        return


class AsyncLogpOp(AsyncOp, LogpOp):
    async def perform_async(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        logp = await self._logp_func(*inputs)
        output_storage[0][0] = logp
        return


class LogpGradOp(Op):
    """An Op that wraps a callable returning a log-potential and its gradient w.r.t the inputs.

    This Op returns the log-potential AND the gradient, but it also
    has a `.grad()` which returns only the gradient.
    """

    _props = ("_logp_grad_func",)

    def __init__(self, logp_grad_func: LogpGradFunc) -> None:
        self._logp_grad_func = logp_grad_func
        super().__init__()

    def make_node(self, *inputs: Union[Variable, int, float, np.ndarray]) -> Apply:
        logp = at.scalar()
        input_tensors = list(map(at.as_tensor, inputs))
        grad = [i.type() for i in input_tensors]
        return Apply(
            op=self,
            inputs=input_tensors,
            outputs=[logp, *grad],
        )

    def perform(
        self,
        node: Apply,
        inputs: Sequence[np.ndarray],
        output_storage: OutputStorageType,
    ) -> None:
        logp, gradient = self._logp_grad_func(*inputs)
        output_storage[0][0] = logp
        for g, grad in enumerate(gradient):
            output_storage[1 + g][0] = grad
        return

    def grad(self, inputs: Sequence[Variable], output_grads: List[Variable]) -> List[Variable]:
        # Unpack the output gradients of which we only need the
        # one w.r.t. logp
        g_logp, *gs_inputs = output_grads
        for i, g in enumerate(gs_inputs):
            if not isinstance(g.type, pytensor.gradient.DisconnectedType):
                raise ValueError(f"Can't propagate gradients wrt parameter {i+1}")
        # Call again on the original inputs, to obtain a handle
        # on the gradient. The computation will not actually be
        # performed again, because this call takes the same inputs
        # as the original one and will be optimized-away.
        _, *gradients = self(*inputs)
        # Return symbolic gradients for each input (of which there is one).
        return [g_logp * g for g in gradients]


class AsyncLogpGradOp(AsyncOp, LogpGradOp):
    async def perform_async(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        logp, gradient = await self._logp_grad_func(*inputs)
        output_storage[0][0] = logp
        for g, grad in enumerate(gradient):
            output_storage[1 + g][0] = grad
        return
