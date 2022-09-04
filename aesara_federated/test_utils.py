from aesara_federated import utils
from aesara_federated.rpc import GetLoadResult


def test_argmin_load():
    assert utils.argmin_none_or_func([], float) is None
    assert utils.argmin_none_or_func([None, None], float) is None
    loads = [
        None,
        GetLoadResult(3, 0.5, 0.2),  # <- min RAM
        None,
        GetLoadResult(2, 0.05, 0.4),  # <- min CPU
        GetLoadResult(1, 0.1, 0.6),  # <- min n_clients
    ]
    assert utils.argmin_none_or_func(loads, lambda l: l.percent_ram) == 1
    assert utils.argmin_none_or_func(loads, lambda l: l.percent_cpu) == 3
    assert utils.argmin_none_or_func(loads, lambda l: l.n_clients) == 4
    pass
