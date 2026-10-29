import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments.audit_instances import analyze
from experiments.audit_regions import run


class InstanceAuditTests(unittest.TestCase):
    def test_empty_labels_and_matching_competition(self):
        empty=np.zeros((2,2),dtype=np.uint32)
        summary,rows=analyze(empty,empty,np.ones_like(empty,dtype=bool))
        self.assertEqual(summary['instance_tp'],0)
        self.assertEqual(rows,[])
        pred=np.ones((2,4),dtype=np.uint32)
        gt=np.array([[1,1,2,2],[1,1,2,2]],dtype=np.uint32)
        summary,rows=analyze(pred,gt,np.ones_like(gt,dtype=bool))
        loser=next(r for r in rows if r['kind']=='gt' and r['status']!='matched')
        self.assertEqual(loser['status'],'one_to_one_competition')
    def test_merge_split_and_missing_objects_are_separate_evidence(self):
        gt=np.array([[1,1,2,2],[3,3,3,3],[4,4,4,4]],dtype=np.uint32)
        pred=np.array([[1,1,1,1],[2,2,3,3],[0,0,0,0]],dtype=np.uint32)
        summary,rows=analyze(pred,gt,np.ones_like(gt,dtype=bool))
        self.assertEqual(summary['predictions_with_merge_evidence'],1)
        self.assertEqual(summary['gt_with_split_evidence'],1)
        missing=next(r for r in rows if r['kind']=='gt' and r['id']==4)
        self.assertEqual(missing['status'],'no_foreground_overlap')
        self.assertEqual(summary['instance_tp'],2)
        self.assertEqual(summary['instance_fn'],2)

    def test_iou_failure_and_roi_cut_flags(self):
        gt=np.zeros((6,6),dtype=np.uint32)
        gt[1:5,1:5]=1
        pred=np.zeros_like(gt)
        pred[1,1]=1
        mask=np.ones_like(gt,dtype=bool)
        mask[4]=False
        summary,rows=analyze(pred,gt,mask,(100,200))
        target=next(r for r in rows if r['kind']=='gt')
        self.assertEqual(target['status'],'below_iou_0.5')
        self.assertTrue(target['roi_or_crop_cut_possible'])
        self.assertEqual(target['x_min'],101)


class SpatialAuditTests(unittest.TestCase):
    def test_missing_provenance_is_pending_without_opening_rasters(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'config.json'
            path.write_text('{}',encoding='utf-8')
            result=run(path,root/'audit')
            self.assertEqual(result['status'],'pending_inputs')
            self.assertFalse(result['training_independence_confirmed'])

    def test_nonoverlap_is_insufficient_when_buffer_is_violated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=dict(reference='train.tif',checkpoint='checkpoint.bin',training_provenance='synthetic fixture',
                        all_training_footprints_declared=True,train_mask='train.tif',
                        validation_mask='validation.tif',test_mask='test.tif',buffer_pixels=1)
            (root/'checkpoint.bin').write_bytes(b'fixture')
            for name,x in [('train',1),('validation',2),('test',6)]:
                data=np.zeros((8,8),dtype=np.uint8)
                data[3,x]=1
                with rasterio.open(root/f'{name}.tif','w',driver='GTiff',width=8,height=8,count=1,
                     dtype='uint8',crs='EPSG:32648',transform=from_origin(0,8,1,1)) as dst:
                    dst.write(data,1)
            path=root/'config.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            result=run(path,root/'failed')
            self.assertEqual(result['status'],'spatial_checks_failed')
            self.assertEqual(result['overlap_pixels']['train_validation'],0)
            self.assertEqual(result['buffer_violations_pixels']['train_validation'],1)
            config['buffer_pixels']=0
            path.write_text(json.dumps(config),encoding='utf-8')
            result=run(path,root/'passed')
            self.assertEqual(result['status'],'spatial_checks_passed')
            self.assertFalse(result['training_independence_confirmed'])
            with rasterio.open(root/'test.tif','r+') as dst:
                data=np.zeros((8,8),dtype=np.uint8)
                data[3,1]=1
                dst.write(data,1)
            result=run(path,root/'overlap')
            self.assertEqual(result['status'],'spatial_checks_failed')
            self.assertEqual(result['overlap_pixels']['train_test'],1)
            with rasterio.open(root/'test.tif','r+') as dst:
                data[3,1]=2
                dst.write(data,1)
            with self.assertRaises(ValueError):
                run(path,root/'nonbinary')
            self.assertFalse((root/'nonbinary').exists())
