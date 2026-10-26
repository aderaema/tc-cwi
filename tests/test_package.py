import unittest
from types import SimpleNamespace

import numpy as np
from ulb_workflow import obstacle_sensitivity

from tc_cwi import tc_cwi


class PackageTest(unittest.TestCase):
    def test_public_api(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * t)
        signals = np.column_stack((y, y))

        baseline = tc_cwi("getccns", y, signals)
        self.assertIsInstance(baseline, SimpleNamespace)
        np.testing.assert_allclose(baseline.CC, 1)
        np.testing.assert_allclose(baseline.DC, 0, atol=1e-12)
        with self.assertRaises(TypeError):
            _ = baseline["CC"]

        for command, field in (("findeps", "DCsf"), ("findeps -ma", "DCma")):
            result = tc_cwi(command, y, signals, t, 0.0, 2)
            self.assertIsInstance(result, SimpleNamespace)
            np.testing.assert_allclose(getattr(result, field), 0, atol=1e-12)
            with self.assertRaises(TypeError):
                _ = result[field]

        single_out = tc_cwi("findeps", y, y, t, 0.0, 2)
        self.assertIsInstance(single_out, SimpleNamespace)
        self.assertAlmostEqual(single_out.eps, 0.0, places=6)
        self.assertAlmostEqual(single_out.DC, 0.0, places=6)
        with self.assertRaises(TypeError):
            _ = single_out["eps"]

        result = obstacle_sensitivity([0.1, 0.2], [0.1, 0.15, 0.8, 0.85])
        self.assertIsInstance(result, SimpleNamespace)
        self.assertEqual(result.insertion_index, 2)
        self.assertEqual(result.sensitivity_ratio, 4)
        with self.assertRaises(TypeError):
            _ = result["insertion_index"]

        for reference_dc, obstacle_dc in (
            ([], [0.1, 0.2]),
            ([0.1], [0.2]),
            ([0], [0.1, 0.2]),
            ([np.nan], [0.1, 0.2]),
            ([0.1], [0.2, np.inf]),
        ):
            with self.assertRaises(ValueError):
                obstacle_sensitivity(reference_dc, obstacle_dc)


if __name__ == "__main__":
    unittest.main()
