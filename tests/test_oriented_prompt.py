import unittest
from types import SimpleNamespace

import numpy as np

from experiments.oriented_prompt import OrientedPredictor,geometry,validate_prompts


class OrientedPromptTests(unittest.TestCase):
    box=[[5,5],[18,11],[3,12]]

    def test_rotation_transforms_prompts_and_restores_logits_without_box_clipping(self):
        class Spy:
            model=SimpleNamespace(mask_threshold=0)
            def set_image(self,rgb): self.shape=rgb.shape[:2]
            def predict(self,point_coords,point_labels,box,**kwargs):
                self.points=point_coords;self.labels=point_labels;self.box=box
                yy,xx=np.indices(self.shape)
                return np.array([xx+2*yy+1.0]),np.array([.95]),None
        spy=Spy();adapter=OrientedPredictor(spy,self.box)
        rgb=np.full((24,26,3),120,dtype=np.uint8);adapter.set_image(rgb)
        pts=np.array([[10,10],[12,3]],dtype=np.float32)
        masks,quality,_=adapter.predict(pts,np.array([1,0]),return_logits=True,multimask_output=True)
        expected=pts@adapter.g['rotation'].T-adapter.offset
        np.testing.assert_allclose(spy.points,expected,atol=1e-5)
        np.testing.assert_equal(spy.labels,[1,0])
        self.assertEqual(masks.shape,(1,24,26));self.assertEqual(quality[0],.95)
        self.assertAlmostEqual(masks[0,10,10],expected[0,0]+2*expected[0,1]+1,places=5)
        # This pixel is outside the human box, but must not be cut off by the adapter.
        self.assertGreater(masks[0,16,10],0)
        self.assertFalse(adapter.audit()['mask_clipped_to_box'])

    def test_box_validation_and_positive_containment(self):
        validate_prompts(self.box,(24,26),[[10,10],[12,3]],[1,0])
        for bad in ([[5,5],[5,5],[3,12]],[[5,5],[18,11],[10,7.31]],[[0,0],[25,23],[25,0]],[[1,2]]):
            with self.assertRaises(ValueError): geometry(bad,(24,26))
        with self.assertRaisesRegex(ValueError,'positive points'):
            validate_prompts(self.box,(24,26),[[12,3]],[1])


if __name__=='__main__': unittest.main()
