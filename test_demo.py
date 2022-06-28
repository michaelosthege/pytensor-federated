import numpy as np

import demo


class TestLinearModel:
    def test_grad(self):
        lm = demo.LinearModelBlackbox(
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
