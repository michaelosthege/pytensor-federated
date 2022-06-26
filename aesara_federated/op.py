from typing import List, Sequence

import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op, OutputStorageType, ParamsInputType

from .service import FederatedLogpOpClient


class FederatedLogpOp(Op):
    """An Op that remotely computes a log-potential and its gradient w.r.t the inputs.

    This Op returns the log-potential AND the gradient, but it also
    has a `.grad()` which returns only the gradient.
    """

    _props = ("_client",)

    def __init__(self, client: FederatedLogpOpClient) -> None:
        self._client = client
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
        logp, gradient = self._client.evaluate(*inputs)
        output_storage[0][0] = logp
        for g, grad in enumerate(gradient):
            output_storage[1 + g][0] = grad
        return

    def grad(self, inputs: Sequence[Variable], output_grads: List[Variable]) -> List[Variable]:
        # Call again on the original inputs, to obtain a handle
        # on the gradient. The computation will not actually be
        # performed again, because this call takes the same inputs
        # as the original one and will be optimized-away.
        _, *gradients = self(*inputs)
        # Return symbolic gradients for each input (of which there is one).
        return gradients
