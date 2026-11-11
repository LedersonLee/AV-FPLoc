"""Analytic controls and corrupt-output rejection, without Torch or model data."""
import unittest
import numpy as np
from analyze_router import router_statistics
from matching import FOV


class RouterAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.target = np.ones((8, 36))
        self.z = {head + suffix: np.ones((8, 11 if head == 'V' else 36))
                  for head in ['V', 'A', 'AV', 'F'] for suffix in ['_mu', '_scale']}
        self.z['weights'] = np.zeros((8, 36, 3))
        self.z['weights'][:, :, 1] = 1

    def test_tied_errors_and_constant_correlation(self):
        r = router_statistics(self.z, self.target)
        self.assertEqual(r['fused']['ray_mae'], 0)
        for d in r['per_direction']:
            self.assertEqual(d['max_weight_expert_is_min_error_fraction'], 1)
            for e in d['experts']:
                self.assertIsNone(e['weight_abs_error_spearman'])

    def test_one_hot_and_uniform_control(self):
        self.z['V_mu'][:] = 3
        self.z['AV_mu'][:] = 5
        r = router_statistics(self.z, self.target)
        self.assertEqual(r['fused']['ray_mae'], 0)
        self.assertAlmostEqual(r['uniform_same_heads_mae'], 2)
        self.assertTrue(all(d['max_weight_selection_excess_mae'] == 0 for d in r['per_direction']))

    def test_corrupt_outputs_rejected(self):
        for key, value in [('F_mu', 2), ('F_scale', 2), ('weights', -1), ('A_scale', 0)]:
            with self.subTest(key=key):
                z = {k: v.copy() for k, v in self.z.items()}
                z[key][:] = value
                with self.assertRaises(AssertionError):
                    router_statistics(z, self.target)
        d = next(i for i in range(36) if i not in FOV)
        self.z['weights'][:, d, :] = [.2, .8, 0]
        with self.assertRaises(AssertionError):
            router_statistics(self.z, self.target)

    def test_rank_correlations_and_independent_moments(self):
        x = np.arange(1, 9) / 10
        self.z['A_mu'][:] = (1 + x)[:, None]
        self.z['A_scale'][:] = x[:, None]
        self.z['AV_mu'][:] = 5
        self.z['weights'][:, :, 1] = (1 - x)[:, None]
        self.z['weights'][:, :, 2] = x[:, None]
        for i in range(8):
            mean = (1 - x[i]) * (1 + x[i]) + x[i] * 5
            variance = (1 - x[i]) * (2*x[i]**2 + (1+x[i]-mean)**2) + x[i] * (2 + (5-mean)**2)
            self.z['F_mu'][i] = mean
            self.z['F_scale'][i] = (variance / 2)**.5
        for d in router_statistics(self.z, self.target)['per_direction']:
            a = next(e for e in d['experts'] if e['expert'] == 'A')
            self.assertAlmostEqual(a['weight_abs_error_spearman'], -1)
            self.assertAlmostEqual(a['scale_abs_error_spearman'], 1)


if __name__ == '__main__':
    unittest.main()
