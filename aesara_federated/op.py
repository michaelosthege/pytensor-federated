from typing import List, Sequence

import aesara
import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op, OutputStorageType, ParamsInputType

from .signatures import LogpGradFunc


class FederatedLogpOp(Op):
    """An Op that wraps a callable returning a log-potential and its gradient w.r.t the inputs.

    This Op returns the log-potential AND the gradient, but it also
    has a `.grad()` which returns only the gradient.
    """

    _props = ("_logp_grad_func",)

    def __init__(self, logp_grad_func: LogpGradFunc) -> None:
        self._logp_grad_func = logp_grad_func
        super().__init__()

    def make_node(self, *inputs: Variable) -> Apply:
        logp = at.scalar()
        grad = [i.type() for i in inputs]
        return Apply(
            op=self,
            inputs=inputs,
            outputs=[logp, *grad],
        )

    def perform(
        self,
        node: Apply,
        inputs: Sequence[np.ndarray],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
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
            if not isinstance(g.type, aesara.gradient.DisconnectedType):
                raise ValueError(f"Can't propagate gradients wrt parameter {i+1}")
        # Call again on the original inputs, to obtain a handle
        # on the gradient. The computation will not actually be
        # performed again, because this call takes the same inputs
        # as the original one and will be optimized-away.
        _, *gradients = self(*inputs)
        # Return symbolic gradients for each input (of which there is one).
        return [g_logp * g for g in gradients]
