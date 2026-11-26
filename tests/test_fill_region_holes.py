import unittest

import numpy as np

from experiments.fill_region_holes import candidates


class HoleCandidateTests(unittest.TestCase):
    def fixture(self):
        data=np.zeros((24,24),dtype=np.uint32);data[2:22,2:22]=353;data[8:11,8:12]=0
        return data,np.ones_like(data,dtype=bool)

    def test_closed_background_only_and_diagonal_connection_is_not_hole(self):
        data,roi=self.fixture();labels,rows,_=candidates(data,roi,353)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['pixels'],12)
        self.assertEqual(int((labels>0).sum()),12)
        # Diagonal background path to exterior must block filling (8-connectivity).
        for i in range(9):data[i,i]=0
        labels,rows,_=candidates(data,roi,353)
        self.assertFalse(rows);self.assertFalse(labels.any())

    def test_roi_cut_and_other_instance_are_protected(self):
        data,roi=self.fixture();roi[9,9]=False
        labels,rows,skipped=candidates(data,roi,353)
        self.assertFalse(rows);self.assertGreater(skipped['outside_roi'],0)
        data,roi=self.fixture();data[7,9]=18
        labels,rows,skipped=candidates(data,roi,353)
        self.assertFalse(rows);self.assertGreater(skipped['other_surrounding_ids'],0)
        data[8:11,8:12]=18
        self.assertFalse(candidates(data,roi,353)[1])


if __name__=='__main__':unittest.main()
