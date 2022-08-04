import aesara
import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Apply, Variable

from . import op


class _MockLogpGradOpClient:
    def __init__(self, fn) -> None:
        self._fn = fn
        super().__init__()

    def evaluate(self, *inputs):
        return self._fn(*inputs)

    def __call__(self, *inputs):
        return self.evaluate(*inputs)


def dummy_quadratic_model(a, b):
    rng = np.random.RandomState(42)
    x = np.array([1, 2, 3])
    y = rng.normal(2 * x**2 + 0.5, scale=0.1)
    pred = a * x**2 + b
    cost = np.sum((pred - y) ** 2)
    grad = [
        np.sum(2 * x**2 * (a * x**2 + b - y)),
        np.sum(2 * (a * x**2 + b - y)),
    ]
    return cost, grad


class TestLogpGradOp:
    def test_init(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = op.LogpGradOp(client)
        assert flop._logp_grad_func is client
        pass

    def test_make_node(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = op.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        apply = flop.make_node(a, b)
        assert isinstance(apply, Apply)
        assert len(apply.inputs) == 2
        assert len(apply.outputs) == 3
        pass

    def test_perform(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = op.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        apply = flop.make_node(a, b)
        inputs = [np.array(1.2), np.array(1.5)]
        output_storage = [[None], [None], [None]]
        flop.perform(apply, inputs, output_storage)
        logp, (da, db) = dummy_quadratic_model(*inputs)
        print(logp, (da, db))
        print(output_storage)
        np.testing.assert_array_equal(output_storage[0][0], logp)
        np.testing.assert_array_equal(output_storage[1][0], da)
        np.testing.assert_array_equal(output_storage[2][0], db)
        pass

    def test_forward(self):
        client = _MockLogpGradOpClient(dummy_quadratic_model)
        flop = op.LogpGradOp(client)
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
        flop = op.LogpGradOp(client)
        a = at.scalar()
        b = at.scalar()
        logp, *_ = flop(a, b)
        ga, gb = at.grad(logp, [a, b])
        fn = aesara.function(inputs=[a, b], outputs=[logp, ga, gb])
        exlogp, (exda, exdb) = dummy_quadratic_model(1.4, 0.5)
        actual = fn(1.4, 0.5)
        np.testing.assert_array_equal(actual[0], exlogp)
        np.testing.assert_array_equal(actual[1], exda)
        np.testing.assert_array_equal(actual[2], exdb)
        pass
