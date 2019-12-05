from unittest import TestCase
import numpy as np
from modules.local_linear_forest import LocalLinearForestRegressor

class TestLocalLinearForest(TestCase):
    def setUp(self):
        self.llf = LocalLinearForestRegressor(n_estimators=5)

    def test_get_forest_coefficients(self):
        self.llf._X_train = np.array([
            [3, 1, 4],
            [3, 4, 2],
            [3, 2, 1],
            [2, 1, 4],
        ])
        self.llf._incidence_matrix = np.array([
            [3, 1, 2, 1, 1],
            [3, 4, 2, 5, 1],
            [3, 2, 1, 4, 4],
            [2, 1, 4, 4, 5],
        ])

        actual_leaf_ids = np.array([[2, 1, 4, 4, 5]])
        coeffs = self.llf._get_forest_coefficients(actual_leaf_ids)

        self.assertTrue(len(coeffs) == self.llf._X_train.shape[0])
        self.assertEqual(coeffs, [0.1, 0, 0.1, 1/5*8/2])