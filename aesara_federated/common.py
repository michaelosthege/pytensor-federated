"""
Wrappers around ArraysToArrays service commonly used signatures for modeling.
"""
from typing import Sequence, Tuple

import numpy as np

from .service import ArraysToArraysServiceClient
from .signatures import ComputeFunc, LogpFunc, LogpGradFunc


def wrap_logp_func(logp_func: LogpFunc) -> ComputeFunc:
    """Wraps a non-differentiable logp function as a ``ComputeFunc``."""

    def compute_func(*inputs):
        logp = logp_func(*inputs)
        if not isinstance(logp, np.ndarray):
            raise TypeError(f"The logp value must be a scalar ndarray. Got {type(logp)} instead.")
        if not logp.shape == ():
            raise Exception(f"Returned logp must be scalar, but got shape {logp.shape}")
        return (logp,)

    return compute_func


def wrap_logp_grad_func(logp_grad_func: LogpGradFunc) -> ComputeFunc:
    """Wraps a logp function that also returns gradients as a ``ComputeFunc``."""

    def compute_func(*inputs):
        result = logp_grad_func(*inputs)
        if not len(result) == 2:
            raise TypeError(
                f"The return value of the logp function must be a tuple"
                " of a scalar ndarray and a list of ndarrays for the gradients."
                f" Got {type(result)} instead."
            )
        logp, gradients = result
        if not isinstance(logp, np.ndarray):
            raise TypeError(f"The logp value must be a scalar ndarray. Got {type(logp)} instead.")
        if not logp.shape == ():
            raise Exception(f"Returned logp should be scalar, but got shape {logp.shape}")
        if not len(gradients) == len(inputs):
            raise Exception(
                "Number of gradients does not match number of inputs."
                f"\ninputs: {inputs}\ngradients: {gradients}"
            )
        return logp, *gradients

    return compute_func


class LogpServiceClient:
    """Wraps the ``ArraysToArraysServiceClient`` in a ``LogpFunc`` signature."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        *,
        hosts_and_ports: Sequence[Tuple[str, int]] = None,
    ) -> None:
        """Wraps the ``ArraysToArraysServiceClient`` in a ``LogpGradFunc`` signature.

        Parameters
        ----------
        host : str
            IP address or host name of the remote gRPC server.
        port : int
            Port of the gRPC server.
        hosts_and_ports : list of (host, port) tuples, optional
            If provided, this takes precedence over `host` and `port`.
            Uses client-side load balancing to choose which server to connec to.
        """
        self._client = ArraysToArraysServiceClient(host, port, hosts_and_ports=hosts_and_ports)
        super().__init__()

    def __call__(self, *inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Alias for ``.evaluate(*inputs)``."""
        return self.evaluate(*inputs)

    def evaluate(
        self, *inputs: Sequence[np.ndarray], use_stream=True
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        """Evaluate the federated blackbox logp gradient function on inputs.

        Parameters
        ----------
        *inputs
            NumPy `ndarray` inputs.
        use_stream : bool
            If ``True`` (default), the RPC is performed through a bidirectional stream,
            which is much faster than sending individual (unary/unary) RPCs.

        Returns
        -------
        logp : ndarray
            The log-potential of the federated blackbox function.
        """
        (logp,) = self._client.evaluate(*inputs, use_stream=use_stream)
        return logp

    async def evaluate_async(
        self, *inputs: Sequence[np.ndarray], use_stream=True
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        (logp,) = await self._client.evaluate_async(*inputs, use_stream=use_stream)
        return logp


class LogpGradServiceClient:
    """Wraps the ``ArraysToArraysServiceClient`` in a ``LogpGradFunc`` signature."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        *,
        hosts_and_ports: Sequence[Tuple[str, int]] = None,
    ) -> None:
        """Wraps the ``ArraysToArraysServiceClient`` in a ``LogpGradFunc`` signature.

        Parameters
        ----------
        host : str
            IP address or host name of the remote gRPC server.
        port : int
            Port of the gRPC server.
        hosts_and_ports : list of (host, port) tuples, optional
            If provided, this takes precedence over `host` and `port`.
            Uses client-side load balancing to choose which server to connec to.
        """
        self._client = ArraysToArraysServiceClient(host, port, hosts_and_ports=hosts_and_ports)
        super().__init__()

    def __call__(self, *inputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        """Alias for ``.evaluate(*inputs)``."""
        return self.evaluate(*inputs)

    def evaluate(
        self, *inputs: Sequence[np.ndarray], use_stream=True
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        """Evaluate the federated blackbox logp gradient function on inputs.

        Parameters
        ----------
        *inputs
            NumPy `ndarray` inputs.
        use_stream : bool
            If ``True`` (default), the RPC is performed through a bidirectional stream,
            which is much faster than sending individual (unary/unary) RPCs.

        Returns
        -------
        logp : ndarray
            The log-potential of the federated blackbox function.
        *gradients
            Sequence of ``ndarray`` with gradients of ``logp`` w.r.t. ``inputs``.
        """
        logp, *gradients = self._client.evaluate(*inputs, use_stream=use_stream)
        return logp, gradients

    async def evaluate_async(
        self, *inputs: Sequence[np.ndarray], use_stream=True
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        logp, *gradients = await self._client.evaluate_async(*inputs, use_stream=use_stream)
        return logp, gradients
