import logging

import arviz
import pymc as pm

from aesara_federated import LogpGradOp, LogpGradServiceClient

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def run_model():
    _log.info("Connecting to remote model")
    client = LogpGradServiceClient("127.0.0.1", port=50051)
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
    run_model()
