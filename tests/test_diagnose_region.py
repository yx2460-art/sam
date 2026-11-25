import unittest

import numpy as np

from experiments.diagnose_region import boundary_errors,changes,instance_matches,residual_components
from experiments.evaluate_tiles import roi_counts


class DiagnoseRegionTests(unittest.TestCase):
    def test_residual_components_include_offsets_adjacency_and_roi_clipping(self):
        data=np.zeros((10,12),dtype=np.uint32);data[2:5,2:4]=2;data[5,4]=2;data[7:9,8:10]=2
        data[2:5,4]=354;mask=np.ones_like(data,dtype=bool);mask[8:]=False
        labels,rows=residual_components(data,mask,2,(100,200))
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['pixels'],7)
        self.assertEqual(rows[0]['global_xyxy'],[102,202,105,206])
        self.assertIn('354',rows[0]['adjacent_ids']);self.assertTrue(rows[1]['touches_roi_edge'])
        self.assertEqual(int((labels>0).sum()),9);self.assertFalse(labels[~mask].any())
        empty,rows=residual_components(data,mask,999)
        self.assertEqual(rows,[]);self.assertFalse(empty.any())
    def test_boundary_maps_match_evaluator_with_holes_and_roi_edges(self):
        gt=np.zeros((30,35),dtype=np.uint32);gt[3:26,4:18]=1;gt[3:26,18:30]=2
        pred=gt.copy();pred[3:26,16:18]=2;pred[8:12,9:11]=0
        mask=np.zeros_like(gt,dtype=bool);mask[2:28,3:32]=True;mask[15:18,12:15]=False
        for tolerance in (0,1,2,3):
            codes,actual=boundary_errors(pred,gt,mask,tolerance)
            expected=roi_counts(pred,gt,mask,tolerance)
            for key,value in actual.items():self.assertEqual(value,expected[key])
            self.assertFalse(codes[~mask].any())
            self.assertEqual(int((codes==2).sum()),actual['boundary_pred']-actual['boundary_matched_pred'])
            self.assertEqual(int((codes==3).sum()),actual['boundary_gt']-actual['boundary_matched_gt'])

    def test_change_types_partition_modified_pixels_without_calling_id_changes_correct(self):
        before=np.array([[0,0,1,1,2,8]],dtype=np.uint32)
        after=np.array([[3,4,0,0,5,9]],dtype=np.uint32)
        gt=np.array([[1,0,1,0,2,8]],dtype=np.uint32);mask=np.array([[1,1,1,1,1,0]],dtype=bool)
        codes,counts=changes(before,after,gt,mask)
        np.testing.assert_equal(codes,[[1,2,3,4,5,0]])
        self.assertTrue(all(value==1 for value in counts.values()))

    def test_matching_list_uses_same_greedy_iou_threshold_as_evaluator(self):
        gt=np.array([[1,1,1,1,2,2],[1,1,1,1,2,2]],dtype=np.uint32)
        pred=np.array([[3,3,4,4,5,5],[3,3,4,4,5,5]],dtype=np.uint32)
        mask=np.ones_like(gt,dtype=bool)
        rows,missed=instance_matches(pred,gt,mask);expected=roi_counts(pred,gt,mask,0)
        self.assertEqual(sum(row['matched_gt'] is not None for row in rows),expected['instance_tp'])
        self.assertAlmostEqual(sum(row['matched_iou'] for row in rows),expected['matched_iou_sum'])
        self.assertEqual(len(missed),expected['instance_fn'])
        empty,missed=instance_matches(np.zeros_like(pred),gt,mask)
        self.assertEqual(empty,[]);self.assertEqual(missed,[1,2])


if __name__=='__main__':unittest.main()
