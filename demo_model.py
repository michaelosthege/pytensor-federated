import argparse
import logging

import arviz
import pymc as pm

from aesara_federated import LogpGradOp, LogpGradServiceClient

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def run_model(*, host: str = "127.0.0.1", port: int = 50051):
    _log.info("Connecting to remote model at %s:%i", host, port)
    client = LogpGradServiceClient(host, port=port)
    _log.info("Wrapping into a LogpGradOp")
    remote_model = LogpGradOp(client)

    with pm.Model() as pmodel:
        intercept = pm.Normal("intercept")
        slope = pm.Normal("slope")
        logp, *_ = remote_model(intercept, slope)
        pm.Potential(
            "potential",
            var=logp,
        )

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
        "--port",
        default=50051,
        help="Port number of the ArraysToArrays gRPC service on the worker node.",
    )
    args, _ = parser.parse_known_args()

    run_model(host=args.host, port=int(args.port))
