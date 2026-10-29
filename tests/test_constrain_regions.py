import unittest

import numpy as np

from experiments.constrain_regions import boundary_band, constrain


class BoundaryReplayTests(unittest.TestCase):
    def test_protects_interior_and_preserves_candidates_inside_fixed_band(self):
        initial=np.zeros((15,15),dtype=np.uint32)
        initial[2:13,2:13]=1
        candidate=initial.copy()
        candidate[7,7]=candidate[2,7]=0
        candidate[1,7]=1
        allowed=boundary_band(initial,1)
        result=constrain(initial,candidate,allowed)
        self.assertEqual(result[7,7],1)
        self.assertEqual(result[2,7],0)
        self.assertEqual(result[1,7],1)
        np.testing.assert_array_equal(result[~allowed],initial[~allowed])
        self.assertEqual(initial[2,7],1)

    def test_instance_boundaries_count_without_artificial_array_border(self):
        labels=np.ones((9,9),dtype=np.uint32)
        self.assertFalse(boundary_band(labels,2).any())
        labels[:,5:]=2
        band=boundary_band(labels,0)
        self.assertTrue(band[:,4:6].all())
        self.assertFalse(band[:,:4].any())
        with self.assertRaises(ValueError):
            boundary_band(labels,-1)
