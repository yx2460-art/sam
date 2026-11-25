import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio

import test_conflict_regions
import test_split_instances
from experiments.run_tiles import run as run_tiles
from experiments.run_tiles import digest
from experiments.region_workflow import prepare,fusion_scores,start,infer,apply


class RegionWorkflowTests(unittest.TestCase):
    def test_same_candidates_can_be_applied_with_foreground_preserving_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            with rasterio.open(root/'prediction.tif') as src:source=src.read(1);profile=src.profile
            source[8:12,8:12]=0;source[16:20,16:20]=2
            with rasterio.open(root/'source_holes.tif','w',**profile) as dst:dst.write(source,1)
            params=json.loads((root/'tiles'/'manifest.json').read_text())['parameters']
            params.update(prediction=str(root/'source_holes.tif'),output_dir=root/'holes_tiles')
            run_tiles(SimpleNamespace(**params),refine=lambda rgb,original,predictor,options:(original.copy(),[]))
            m=prepare(root/'holes_tiles',root/'region',bounds=[4,4,16,16],context=2)
            first=fusion_scores(root/'region',args.gt,root/'scores')
            start(root/'region',root/'scores','union','control',root/'interaction',target_id=1,interaction_mode='single-target')
            points=dict(project_sha256=digest(root/'interaction'/'project.json'),interaction_mode='single-target',
                instances=[dict(id=1,points=[[5,5]],point_labels=[1])])
            (root/'points.json').write_text(json.dumps(points))
            proposals=infer(root/'interaction',root/'points.json',args.checkpoint,root/'proposals',predictor=test_split_instances.Predictor(True))
            proposal_hash=digest(root/'proposals'/'proposals.json')
            (root/'approve.json').write_text(json.dumps(dict(proposals_sha256=proposal_hash,accepted_ids=[1])))
            normal=apply(root/'proposals',root/'approve.json',root/'normal')
            actual=rasterio.open
            def no_gt(path,*a,**kw):
                self.assertNotEqual(Path(path).resolve(),args.gt.resolve())
                return actual(path,*a,**kw)
            with patch('rasterio.open',side_effect=no_gt):
                control=apply(root/'proposals',root/'approve.json',root/'control',preserve_foreground=True)
            self.assertEqual(normal['added_foreground_pixels'],16)
            self.assertEqual(control['suppressed_added_foreground_pixels'],16)
            self.assertEqual(control['added_foreground_pixels'],0);self.assertEqual(control['removed_foreground_pixels'],0)
            self.assertGreater(control['changed_pixels'],0)
            self.assertEqual(proposal_hash,digest(root/'proposals'/'proposals.json'))
            with actual(m['candidates']['union']) as src:before=src.read(1)
            with actual(m['mask']) as src:mask=src.read(1)>0
            with actual(root/'control'/'region_result.tif') as src:after=src.read(1)
            np.testing.assert_equal(before>0,after>0)
            np.testing.assert_equal(before[~mask],after[~mask])
            np.testing.assert_equal(before[before==2],after[before==2])
            final=fusion_scores(root/'region',args.gt,root/'control_eval',root/'control')
            self.assertEqual(final['write_policy'],'preserve-foreground')
            for metric in ('mask_iou','mask_precision','mask_recall','mask_dice'):
                self.assertEqual(final['evaluation']['interactive'][metric],first['evaluation']['union'][metric])
            from experiments.diagnose_region import diagnose
            diagnosis=diagnose(root/'region',root/'control',args.gt,root/'diagnose_control')
            self.assertEqual(diagnosis['last_round_changes']['wrong_added_foreground'],0)
            self.assertEqual(diagnosis['last_round_changes']['correct_added_foreground'],0)

    def test_oriented_box_inference_exports_original_grid_and_same_roi(self):
        class BoxPredictor:
            model=SimpleNamespace(mask_threshold=0)
            def set_image(self,rgb): self.shape=rgb.shape[:2]
            def predict(self,point_coords,point_labels,box,**kwargs):
                assert box is not None
                np.testing.assert_equal(point_labels,[1,0])
                yy,xx=np.indices(self.shape)
                raw=np.where((xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3]),5.0,-5.0)
                return np.array([raw]),np.array([.95]),None
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            prepare(root/'tiles',root/'region',bounds=[4,4,16,16],context=2)
            first=fusion_scores(root/'region',args.gt,root/'scores')
            start(root/'region',root/'scores','union','rotated target',root/'interaction',interaction_mode='single-target')
            project=json.loads((root/'interaction'/'project.json').read_text());anchor=project['regions'][0]['id']
            points=dict(project_sha256=digest(root/'interaction'/'project.json'),interaction_mode='single-target',
                instances=[dict(id=anchor,points=[[8,7],[8,14]],point_labels=[1,0],oriented_box=[[3,3],[16,7],[2,8]])])
            (root/'points.json').write_text(json.dumps(points))
            proposals=infer(root/'interaction',root/'points.json',args.checkpoint,root/'proposals',predictor=BoxPredictor())
            p=proposals['proposals'][0];self.assertEqual(p['status'],'proposal')
            self.assertEqual(p['oriented_box']['mode'],'rotated-image-box')
            (root/'approve.json').write_text(json.dumps(dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[anchor])))
            apply(root/'proposals',root/'approve.json',root/'result')
            with rasterio.open(root/'region'/'union.tif') as src:before=src.read(1);transform=src.transform
            with rasterio.open(root/'region'/'region_mask.tif') as src:mask=src.read(1)>0
            with rasterio.open(root/'result'/'region_result.tif') as src:
                self.assertEqual(src.transform,transform);after=src.read(1)
            self.assertNotEqual(after[7,8],before[7,8]);self.assertEqual(after[14,8],before[14,8])
            np.testing.assert_equal(before[~mask],after[~mask])
            final=fusion_scores(root/'region',args.gt,root/'final',root/'result')
            self.assertEqual(first['evaluation']['union'],final['evaluation']['union'])
            from experiments.diagnose_region import diagnose
            before_hash=digest(root/'result'/'region_result.tif')
            diagnosis=diagnose(root/'region',root/'result',args.gt,root/'diagnose')
            self.assertEqual(diagnosis['evaluation'],final['evaluation'])
            self.assertEqual(before_hash,digest(root/'result'/'region_result.tif'))
            self.assertTrue((root/'diagnose'/'residual_overview.png').exists())
            self.assertEqual(diagnosis['residual']['target_id'],anchor)
            self.assertEqual(sum(row['pixels'] for row in diagnosis['residual']['components']),diagnosis['residual']['roi_pixels'])
            self.assertTrue((root/'diagnose'/'index.html').exists())
            self.assertTrue(diagnosis['new_pred_ids'])

    def test_joint_prompts_reach_sam_and_keep_one_child_through_export(self):
        class JointPredictor:
            model=SimpleNamespace(mask_threshold=0)
            def set_image(self,rgb): self.shape=rgb.shape[:2]
            def predict(self,point_coords,point_labels,box,**kwargs):
                np.testing.assert_equal(point_labels,[0,1,1])
                np.testing.assert_equal(point_coords,[[5,14],[5,5],[6,6]])
                self.called=True
                assert box is None
                bad=np.full(self.shape,5.0)
                good=np.full(self.shape,-5.0);good[2:10,2:10]=5
                return np.array([bad,good]),np.array([.99,.95]),None
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            prepare(root/'tiles',root/'region',bounds=[4,4,16,16],context=2)
            fusion_scores(root/'region',args.gt,root/'scores')
            start(root/'region',root/'scores','union','single point failed',root/'interaction',interaction_mode='single-target')
            project=json.loads((root/'interaction'/'project.json').read_text());anchor=project['regions'][0]['id']
            points=dict(project_sha256=digest(root/'interaction'/'project.json'),interaction_mode='single-target',
                instances=[dict(id=anchor,points=[[5,14],[5,5],[6,6]],point_labels=[0,1,1])])
            (root/'points.json').write_text(json.dumps(points))
            predictor=JointPredictor()
            proposals=infer(root/'interaction',root/'points.json',args.checkpoint,root/'proposals',predictor=predictor)
            self.assertTrue(predictor.called)
            p=proposals['proposals'][0];self.assertEqual(p['status'],'proposal')
            self.assertEqual(len(p['child_ids']),1)
            self.assertIn('negative_included',p['detail']['diagnostics'][0]['candidates'][0]['rejection_codes'])
            self.assertEqual(p['detail']['children'][0]['candidate'],1)
            (root/'approve.json').write_text(json.dumps(dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[anchor])))
            apply(root/'proposals',root/'approve.json',root/'result')
            with rasterio.open(root/'region'/'union.tif') as src:before=src.read(1)
            with rasterio.open(root/'result'/'region_result.tif') as src:after=src.read(1)
            self.assertEqual(before[14,5],after[14,5]);self.assertNotEqual(before[5,5],after[5,5])
            fusion_scores(root/'region',args.gt,root/'final',root/'result')
            # A second parcel is extracted from the last approved raster, not from union.
            start(root/'region',root/'scores','union','next parcel',root/'round2',target_id=anchor,
                  interaction_mode='single-target',from_result=root/'result')
            second_project=json.loads((root/'round2'/'project.json').read_text())
            self.assertEqual(Path(second_project['instances']),root/'result'/'region_result.tif')
            self.assertGreaterEqual(second_project['max_source_id'],max(p['child_ids']))
            class SecondPredictor(JointPredictor):
                def predict(self,point_coords,point_labels,box,**kwargs):
                    np.testing.assert_equal(point_labels,[1,0])
                    assert box is None
                    raw=np.full(self.shape,-5.0);raw[11:18,2:10]=5
                    return np.array([raw]),np.array([.95]),None
            second_points=dict(project_sha256=digest(root/'round2'/'project.json'),interaction_mode='single-target',
                instances=[dict(id=anchor,points=[[5,14],[5,5]],point_labels=[1,0])])
            (root/'second_points.json').write_text(json.dumps(second_points))
            second=infer(root/'round2',root/'second_points.json',args.checkpoint,root/'second_proposals',predictor=SecondPredictor())
            second_ids=second['proposals'][0]['child_ids']
            self.assertGreater(min(second_ids),max(p['child_ids']))
            (root/'second_approve.json').write_text(json.dumps(dict(proposals_sha256=digest(root/'second_proposals'/'proposals.json'),accepted_ids=[anchor])))
            second_result=apply(root/'second_proposals',root/'second_approve.json',root/'second_result')
            self.assertEqual(second_result['interaction_round'],2)
            with rasterio.open(root/'second_result'/'region_result.tif') as src:after_second=src.read(1)
            np.testing.assert_equal(after_second[np.isin(after,p['child_ids'])],after[np.isin(after,p['child_ids'])])
            self.assertEqual(after_second[14,5],second_ids[0])
            final=fusion_scores(root/'region',args.gt,root/'second_final',root/'second_result')
            self.assertEqual(final['interaction_round'],2)
            self.assertIn('previous_interactive',final['evaluation'])
            from experiments.diagnose_region import diagnose
            diagnosis=diagnose(root/'region',root/'second_result',args.gt,root/'diagnose_second')
            self.assertEqual(diagnosis['last_round_reference'],'previous_interactive')
            self.assertEqual(diagnosis['evaluation'],final['evaluation'])
            (root/'empty_approve.json').write_text(json.dumps(dict(proposals_sha256=digest(root/'second_proposals'/'proposals.json'),accepted_ids=[])))
            with self.assertRaisesRegex(ValueError,'No candidate accepted'):
                apply(root/'second_proposals',root/'empty_approve.json',root/'empty_result')
            self.assertFalse((root/'empty_result').exists())
            with self.assertRaisesRegex(ValueError,'original fusion method'):
                start(root/'region',root/'scores','intersection','wrong baseline',root/'wrong_baseline',from_result=root/'result')
            points['instances'][0]['point_labels']=[0,0,0]
            (root/'invalid.json').write_text(json.dumps(points))
            with self.assertRaisesRegex(ValueError,'at least one positive'):
                infer(root/'interaction',root/'invalid.json',args.checkpoint,root/'invalid',predictor=predictor)

    def test_fusion_scores_then_interaction_use_fused_source_and_same_region(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            original_bytes=(root/'prediction.tif').read_bytes();actual=rasterio.open
            def no_gt(path,*a,**kw):
                self.assertNotEqual(Path(path).resolve(),args.gt.resolve())
                return actual(path,*a,**kw)
            experiment=root/'region'
            with patch('rasterio.open',side_effect=no_gt):
                m=prepare(root/'tiles',experiment,bounds=[4,4,16,16],context=2)
            self.assertEqual(m['scoring_pixels'],256)
            first=fusion_scores(experiment,args.gt,root/'fusion_eval')
            self.assertEqual(set(first['evaluation']),{'original','center','union','intersection'})
            with patch('rasterio.open',side_effect=no_gt):
                start(experiment,root/'fusion_eval','union','visually merged despite fusion',root/'interaction')
                project=json.loads((root/'interaction'/'project.json').read_text())
                self.assertEqual(project['instances'],m['candidates']['union'])
                anchor=project['regions'][0]['id']
                points=dict(project_sha256=digest(root/'interaction'/'project.json'),instances=[dict(id=anchor,points=[[8,5],[8,14]])])
                (root/'points.json').write_text(json.dumps(points))
                proposals=infer(root/'interaction',root/'points.json',args.checkpoint,root/'proposals',predictor=test_split_instances.Predictor())
                self.assertEqual(proposals['prompt_mode'],'point-only')
                self.assertEqual(proposals['proposals'][0]['status'],'proposal')
                approved=dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[anchor])
                (root/'approval.json').write_text(json.dumps(approved))
                result=apply(root/'proposals',root/'approval.json',root/'result')
                with actual(m['mask']) as src:mask=src.read(1)>0
                with actual(m['candidates']['union']) as src:before=src.read(1)
                with actual(root/'result'/'region_result.tif') as src:after=src.read(1)
                np.testing.assert_equal(before[~mask],after[~mask])
                self.assertGreater(result['changed_pixels'],0)
            final=fusion_scores(experiment,args.gt,root/'post_eval',root/'result')
            for name in first['evaluation']:
                self.assertEqual(first['evaluation'][name],final['evaluation'][name])
            self.assertIn('interactive',final['evaluation'])
            self.assertEqual(final['scoring_pixels'],256)
            self.assertEqual(original_bytes,(root/'prediction.tif').read_bytes())
            with rasterio.open(root/'gt.tif','r+') as gt:
                a=gt.read(1);a[4,4]=8;gt.write(a,1)
            with self.assertRaisesRegex(ValueError,'GT differs'):
                fusion_scores(experiment,args.gt,root/'wrong_gt',root/'result')

    def test_first_scores_required_and_dark_context_points_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            m=prepare(root/'tiles',root/'region',bounds=[4,4,8,8],context=2)
            with self.assertRaises(FileNotFoundError):
                start(root/'region',root/'missing','union','merged',root/'blocked')
            fusion_scores(root/'region',args.gt,root/'scores')
            start(root/'region',root/'scores','intersection','merged',root/'interaction')
            p=json.loads((root/'interaction'/'project.json').read_text())
            points=dict(project_sha256=digest(root/'interaction'/'project.json'),instances=[dict(id=p['regions'][0]['id'],points=[[0,0]])])
            (root/'points.json').write_text(json.dumps(points))
            with self.assertRaisesRegex(ValueError,'outside the fixed'):
                infer(root/'interaction',root/'points.json',args.checkpoint,root/'invalid',predictor=test_split_instances.Predictor())

    def test_empty_fusion_can_start_single_background_point_and_export_fixed_region(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            params=json.loads((root/'tiles'/'manifest.json').read_text())['parameters']
            params['output_dir']=root/'empty_tiles'
            run_tiles(SimpleNamespace(**params),refine=lambda rgb,original,predictor,options:(np.zeros_like(original),[]))
            m=prepare(root/'empty_tiles',root/'region',bounds=[4,4,8,8],context=2)
            fusion_scores(root/'region',args.gt,root/'scores')
            workflow=start(root/'region',root/'scores','intersection','fusion removed parcel foreground',root/'interaction')
            self.assertEqual(workflow['target_id'],0)
            points=dict(project_sha256=digest(root/'interaction'/'project.json'),instances=[dict(id=0,points=[[5,5]])])
            (root/'points.json').write_text(json.dumps(points))
            proposals=infer(root/'interaction',root/'points.json',args.checkpoint,root/'proposals',predictor=test_split_instances.Predictor())
            self.assertEqual(proposals['proposals'][0]['point_roles'][0]['kind'],'background')
            self.assertEqual(proposals['proposals'][0]['status'],'proposal')
            approval=dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[0])
            (root/'approval.json').write_text(json.dumps(approval))
            apply(root/'proposals',root/'approval.json',root/'result')
            with rasterio.open(root/'result'/'region_result.tif') as src:data=src.read(1)
            with rasterio.open(m['mask']) as src:mask=src.read(1)>0
            self.assertTrue(np.any(data[mask]>0));self.assertTrue(np.all(data[~mask]==0))
            final=fusion_scores(root/'region',args.gt,root/'final',root/'result')
            self.assertGreater(final['evaluation']['interactive']['mask_iou'],final['evaluation']['intersection']['mask_iou'])
            control=apply(root/'proposals',root/'approval.json',root/'control',preserve_foreground=True)
            self.assertEqual(control['changed_pixels'],0)
            self.assertGreater(control['suppressed_added_foreground_pixels'],0)


if __name__=='__main__': unittest.main()
