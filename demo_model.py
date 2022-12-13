import argparse
import logging
from typing import Sequence

import arviz
import numpy as np
import pymc as pm

from pytensor_federated import AsyncLogpGradOp, LogpGradOp, LogpGradServiceClient

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def run_model(host: str, ports: Sequence[int], use_async: bool = True):
    _log.info("Connecting to remote model on %s with %i ports.", host, len(ports))
    client = LogpGradServiceClient(hosts_and_ports=[(host, p) for p in ports])
    _log.info("Parallelization: %s", use_async)
    if use_async:
        remote_model = AsyncLogpGradOp(client.evaluate_async)
    else:
        remote_model = LogpGradOp(client.evaluate)

    # We're building a multilevel linear regression model.
    # The remote datasets are identical, so we're just going to offset the intercepts here.
    N = 3

    with pm.Model() as pmodel:
        intercept_mu = pm.Normal("intercept_mu")
        intercept = pm.Normal("intercept", intercept_mu, sigma=0.1, size=N)
        slope = pm.Normal("slope")

        # Run multiple parallelizable forward passes
        for i, off in enumerate(np.linspace(-N / 2, N / 2, N)):
            logp, *_ = remote_model(intercept[i] + off, slope)
            pm.Potential(f"potential_{i}", var=logp)

        _log.info("Running MAP estimation")
        map_ = pm.find_MAP()
        print(map_)

        idata = pm.sample(tune=500, draws=200)

        print(arviz.summary(idata))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a toy model as a worker node.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Hostname or IP address of the worker node."
    )
    parser.add_argument(
        "--ports",
        default=",".join(map(str, range(50000, 50015))),
        type=str,
        help="Port numbers of the ArraysToArrays gRPC service workers.",
    )
    parser.add_argument(
        "--parallel",
        default="true",
        choices=["true", "false"],
        type=str,
        help="Wether to use asynchronous Ops that can parallelize.",
    )
    args, _ = parser.parse_known_args()

    run_model(
        host=args.host,
        ports=[int(p) for p in args.ports.split(",")],
        use_async=args.parallel.lower() == "true",
    )
