import numpy as np

from aesara_federated.npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from aesara_federated.rpc import InputArrays, OutputArrays

from . import service


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
