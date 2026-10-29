import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments.tiles import fuse_result, source_ids, windows
from experiments.run_tiles import run
from experiments.evaluate_tiles import evaluate_run
from experiments.refuse_tiles import refusion


class TileTests(unittest.TestCase):
    def test_grid_covers_edges_without_shifting_origin(self):
        for size,stride in [(4,4),(4,3),(4,2),(16,8)]:
            count=np.zeros((9,11),dtype=int)
            for w in windows(11,9,size,stride,(0,0,11,9)):
                x,y=int(w.col_off),int(w.row_off)
                self.assertEqual(x%stride,0)
                self.assertEqual(y%stride,0)
                count[y:y+int(w.height),x:x+int(w.width)]+=1
            self.assertTrue(np.all(count>0))
        with self.assertRaises(ValueError):
            list(windows(10,10,4,5,(0,0,10,10)))

    def test_fusion_and_stable_ids(self):
        original=np.array([[10,10,20]],dtype=np.uint32)
        local=np.array([[1,1,2]])
        np.testing.assert_equal(source_ids(local,original),original)
        with self.assertRaises(ValueError):
            source_ids(np.ones_like(local),original)
        count=np.array([[2,2,2]])
        votes=np.array([[0,1,2]])
        for mode,expected in [('union',[[0,10,20]]),('intersection',[[0,0,20]]),
                              ('majority',[[0,0,20]]),('center',[[10,0,20]])]:
            result=fuse_result(original,votes,count,np.array([[1,0,1]],dtype=bool),mode)
            np.testing.assert_equal(result,expected)

    def fixture(self,root,max_windows=0):
        profile=dict(driver='GTiff',width=12,height=12,count=1,dtype='uint16',
                     crs='EPSG:32648',transform=from_origin(100,200,1,1))
        labels=np.zeros((12,12),dtype=np.uint16)
        labels[2:10,2:10]=17
        for name in ('image','prediction','coverage','mask','gt'):
            cfg=profile.copy()
            data=labels[None]
            if name=='image':
                cfg.update(count=3,dtype='uint8')
                data=np.full((3,12,12),100,dtype=np.uint8)
            if name in ('coverage','mask'):
                data=np.ones((1,12,12),dtype=np.uint16)
            with rasterio.open(root/f'{name}.tif','w',**cfg) as dst:
                dst.write(data)
        (root/'weights.pth').write_bytes(b'fake-checkpoint-no-model-loaded')
        args=SimpleNamespace(image=root/'image.tif',prediction=root/'prediction.tif',
                             coverage=root/'coverage.tif',eval_mask=root/'mask.tif',
                             checkpoint=root/'weights.pth',output_dir=root/'run',gt=root/'gt.tif',
                             tile_size=8,stride=4,context=0,max_windows=max_windows,
                             fusion='center',model_type='vit_b',device='cpu',rgb_bands=[1,2,3],
                             rgb_range=None,allow_crs_mismatch=False,boundary_tolerance=0)
        return args,labels

    @staticmethod
    def identity(rgb,original,predictor,options):
        return original.copy(),[]

    def test_complete_integration_no_seams_duplicate_ids_and_georeference(self):
        with tempfile.TemporaryDirectory() as folder:
            args,labels=self.fixture(Path(folder))
            result=run(args,refine=self.identity)
            self.assertEqual(result['status'],'complete')
            with rasterio.open(args.output_dir/'instance.tif') as src:
                np.testing.assert_equal(src.read(1),labels)
                self.assertEqual(src.transform,from_origin(100,200,1,1))
            summary=json.loads((args.output_dir/'summary.json').read_text())
            self.assertEqual(summary['sam']['pq'],1)
            self.assertEqual(summary['sam']['counts']['instance_tp'],1)
            with rasterio.open(args.output_dir/'conflict.tif') as src:
                self.assertEqual(int(src.read(1).sum()),0)

    def test_smoke_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as folder:
            args,_=self.fixture(Path(folder),max_windows=1)
            run(args,refine=self.identity)
            self.assertFalse((args.output_dir/'summary.json').exists())
            with self.assertRaises(ValueError):
                evaluate_run(args.output_dir,args.gt,args.eval_mask,False,0)

    def test_cached_refusion_preserves_source_and_reproduces_center(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args,_=self.fixture(root)
            calls=[0]
            def alternating(rgb,original,predictor,options):
                calls[0]+=1
                return (original.copy() if calls[0]%2 else np.zeros_like(original)),[]
            run(args,refine=alternating)
            before={p:p.read_bytes() for p in args.output_dir.rglob('*') if p.is_file()}
            model_calls=calls[0]
            result=refusion(args.output_dir,root/'fusion',args.gt)
            self.assertEqual(result['status'],'complete')
            self.assertFalse(result['sam_inference_performed'])
            self.assertEqual(calls[0],model_calls)
            arrays={}
            for method in ('center','union','intersection','majority'):
                with rasterio.open(root/'fusion'/method/'instance.tif') as src:
                    arrays[method]=src.read(1)
                self.assertTrue((root/'fusion'/method/'summary.json').exists())
                self.assertTrue(set(np.unique(arrays[method])).issubset({0,17}))
            with rasterio.open(args.output_dir/'instance.tif') as src:
                np.testing.assert_equal(arrays['center'],src.read(1))
            self.assertGreater(int((arrays['union']>0).sum()),int((arrays['intersection']>0).sum()))
            self.assertTrue(np.all((arrays['intersection']>0)<=(arrays['majority']>0)))
            self.assertTrue(np.all((arrays['majority']>0)<=(arrays['union']>0)))
            self.assertGreater(result['stats']['scoring_conflict_pixels'],0)
            self.assertTrue((root/'fusion'/'comparison.csv').exists())
            for path,content in before.items():
                self.assertEqual(path.read_bytes(),content)

    def test_refusion_rejects_invalid_votes_and_missing_cache(self):
        for defect in ('votes','missing'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                args,_=self.fixture(root)
                run(args,refine=self.identity)
                if defect=='missing':
                    (args.output_dir/'work'/'center.bin').unlink()
                else:
                    array=np.memmap(args.output_dir/'work'/'votes.bin',mode='r+',dtype='uint32',shape=(12,12))
                    array[0,0]=999
                    array.flush()
                    del array
                with self.assertRaises(ValueError):
                    refusion(args.output_dir,root/'fusion')
                self.assertFalse((root/'fusion').exists())

    def test_refusion_rejects_smoke_or_changed_baseline(self):
        for defect in ('smoke','baseline'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                args,_=self.fixture(root,max_windows=1 if defect=='smoke' else 0)
                run(args,refine=self.identity)
                if defect=='baseline':
                    with rasterio.open(args.prediction,'r+') as dst:
                        dst.write(np.zeros((12,12),dtype='uint16'),1)
                with self.assertRaises(ValueError):
                    refusion(args.output_dir,root/'fusion')
                self.assertFalse((root/'fusion').exists())


if __name__=='__main__':
    unittest.main()
