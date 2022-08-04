import aesara.tensor as at
import numpy as np
import pymc as pm

import demo_node
from aesara_federated.op import LogpGradOp


class TestLinearModel:
    def test_grad(self):
        lm = demo_node.LinearModelBlackbox(
            data_x=[-1, 0, 1],
            data_y=[1, 1, 1],
            sigma=1,
        )
        # Perfect fit
        np.testing.assert_array_equal(
            lm(1, 0)[1],
            [0, 0],
        )
        # Intercept too high
        np.testing.assert_almost_equal(
            lm(1.1, 0)[1],
            [-0.3, 0],
        )
        pass


def test_linear_model_equivalence():
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 1])
    sigma = 0.2

    # Build a log-probability graph of a linear model
    intercept = at.scalar()
    slope = at.scalar()
    mu = intercept + slope * x
    pred = pm.Normal.dist(mu, sigma)
    L_model = pm.joint_logp(pred, y, sum=True)

    # Build the same log-probability using the blackbox Op
    lmb = demo_node.LinearModelBlackbox(x, y, sigma)
    lmbop = LogpGradOp(lmb)
    L_federated = lmbop(intercept, slope)[0]

    # Compare both
    test_point = {
        intercept: 1.2,
        slope: 0.8,
    }
    np.testing.assert_array_equal(
        L_model.eval(test_point),
        L_federated.eval(test_point),
    )

    # And now the gradient
    dL_model = at.grad(L_model, [intercept, slope])
    dL_federated = at.grad(L_federated, [intercept, slope])
    for dM, dF in zip(dL_model, dL_federated):
        np.testing.assert_array_equal(
            dM.eval(test_point),
            dF.eval(test_point),
        )
    pass


def test_linear_model_logp_dlogp_findmap():
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 1])
    sigma = 0.2

    # Build a standard linear model
    with pm.Model() as pmodel:
        intercept = pm.Normal("intercept")
        slope = pm.Normal("slope")
        mu = intercept + slope * x
        pm.Normal("L", mu, sigma, observed=y)

        P1 = pm.compile_fn(pmodel.logp())
        dP1 = pm.compile_fn(pmodel.dlogp())
        map_1 = pm.find_MAP()

    # Build the same model using the blackbox Op and a Potential
    lmb = demo_node.LinearModelBlackbox(x, y, sigma)
    lmbop = LogpGradOp(lmb)
    with pm.Model() as pmodel:
        intercept = pm.Normal("intercept")
        slope = pm.Normal("slope")
        logp, *_ = lmbop(intercept, slope)
        pm.Potential("L", logp)

        P2 = pm.compile_fn(pmodel.logp())
        dP2 = pm.compile_fn(pmodel.dlogp())
        map_2 = pm.find_MAP()

    # Compare the model's logp values
    test_point = dict(intercept=0.5, slope=1.2)
    assert P1(test_point) == P2(test_point)

    # And their gradients
    grad1 = dP1(test_point)
    grad2 = dP2(test_point)
    for dp1, dp2 in zip(grad1, grad2):
        assert dp1 == dp2

    # And also the MAP estimates
    for vname in ["intercept", "slope"]:
        np.testing.assert_almost_equal(map_1[vname], map_2[vname])
    pass
