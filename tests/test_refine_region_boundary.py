import unittest

import numpy as np

from experiments.refine_region_boundary import refine


class BoundaryRefineTests(unittest.TestCase):
    def test_add_only_real_grabcut_accepts_single_positive_and_never_deletes(self):
        rgb=np.full((32,32,3),20,dtype=np.uint8);rgb[8:24,10:21]=220
        source=np.zeros((32,32),dtype=np.uint32);source[8:24,10:21]=7;source[12:19,14:17]=0;source[14:16,17:22]=0;source[:,26:]=12
        roi=np.ones_like(source,dtype=bool)
        after,editable=refine(rgb,source,roi,7,[11,10,23,21],[[15,15]],[1],width=4,edit_mode='add-only')
        self.assertEqual(after[15,15],7)
        np.testing.assert_equal(after[source>0],source[source>0])
        np.testing.assert_equal(after[~editable],source[~editable])
        self.assertEqual(after[6,6],0)
        def isolated(rgb,gc):gc[:]=0;gc[15,15]=1
        with self.assertRaisesRegex(ValueError,'violates an explicit prompt'):
            refine(rgb,source,roi,7,[11,10,23,21],[[15,15]],[1],width=4,edit_mode='add-only',cutter=isolated)
        with self.assertRaisesRegex(ValueError,'negative point on existing foreground'):
            refine(rgb,source,roi,7,[11,10,23,21],[[15,15],[13,15]],[1,0],width=4,edit_mode='add-only')

    def test_real_grabcut_respects_seeds_protected_ids_and_local_scope(self):
        rgb=np.full((32,32,3),20,dtype=np.uint8);rgb[8:24,10:21]=220
        source=np.zeros((32,32),dtype=np.uint32);source[8:24,9:21]=7;source[:,26:]=12
        roi=np.ones_like(source,dtype=bool);roi[:3]=False
        after,editable=refine(rgb,source,roi,7,[6,5,24,27],[[12,15],[9,15]],[1,0],width=3)
        self.assertEqual(after[15,12],7);self.assertEqual(after[15,9],0)
        np.testing.assert_equal(after[~editable],source[~editable])
        np.testing.assert_equal(after[source==12],source[source==12])
        self.assertTrue(np.all(np.isin(after,[0,7,12])))

    def test_invalid_or_protected_prompts_rejected(self):
        rgb=np.zeros((20,20,3),dtype=np.uint8);source=np.zeros((20,20),dtype=np.uint32);source[4:16,4:16]=1;source[:,18:]=2
        roi=np.ones_like(source,dtype=bool)
        for points,labels in [([[5,5]],[1]),([[5,5],[18,5]],[1,0]),([[5,5],[5,5]],[1,0])]:
            with self.assertRaises(ValueError):refine(rgb,source,roi,1,[0,0,20,20],points,labels,width=2)


if __name__=='__main__':unittest.main()
