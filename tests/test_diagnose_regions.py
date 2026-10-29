import unittest

import numpy as np

from experiments.diagnose_regions import transitions


class TransitionTests(unittest.TestCase):
    def test_distinguishes_gross_changes_from_net_losses_and_id_changes(self):
        before=np.array([[1,1,0,0,1,1]])
        after=np.array([[0,0,1,1,2,0]])
        truth=np.array([[1,0,1,0,1,1]])
        mask=np.array([[1,1,1,1,1,0]],dtype=bool)
        result=transitions(before,after,truth,mask)
        self.assertEqual(result,dict(pixels=5,removed_true_foreground=1,
                                    removed_false_foreground=1,added_true_foreground=1,
                                    added_false_foreground=1,relabeled_foreground=1))
