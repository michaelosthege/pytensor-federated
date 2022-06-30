import numpy as np

from aesara_federated.npproto.utils import ndarray_from_numpy, ndarray_to_numpy
from aesara_federated.rpc import FederatedLogpOpInput, FederatedLogpOpOutput

from . import service


def test_compute_function():
    def perform_fun(a, b):
        cost = np.sum(a + b)
        grads = [
            np.prod(a + b),
            np.prod(a - b),
        ]
        return cost, grads

    # Assembly the input protobuf
    a = np.array([1, 2])
    b = np.array([3, 4.5])
    floi = FederatedLogpOpInput(
        inputs=[
            ndarray_from_numpy(a),
            ndarray_from_numpy(b),
        ]
    )
    floo = service._compute_federated_logp(floi, perform_fun)

    # Assert on the returned result protobuf
    assert isinstance(floo, FederatedLogpOpOutput)
    np.testing.assert_array_equal(ndarray_to_numpy(floo.log_potential), 10.5)
    np.testing.assert_array_equal(ndarray_to_numpy(floo.gradients[0]), 4 * 6.5)
    np.testing.assert_array_equal(ndarray_to_numpy(floo.gradients[1]), -2 * -2.5)
    pass
