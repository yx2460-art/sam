import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments.conflict_regions import plan_regions,random_controls,region_mask
from experiments.refine_conflicts import run as run_conflicts
from experiments.run_tiles import run as run_tiles
from experiments.refuse_tiles import refusion


class RegionTests(unittest.TestCase):
    def test_plans_and_random_masks_match_budget_and_are_reproducible(self):
        conflict=np.zeros((32,32),dtype=bool)
        conflict[3,3]=conflict[20,20]=True
        domain=np.ones_like(conflict)
        regions,total=plan_regions(conflict,domain,8,1,1,2)
        target=region_mask(regions,domain.shape)
        controls=random_controls(regions,domain,target,np.ones_like(domain),41)
        again=random_controls(regions,domain,target,np.ones_like(domain),41)
        control_mask=region_mask(controls,domain.shape)
        self.assertEqual(total,2)
        self.assertEqual(target.sum(),control_mask.sum())
        self.assertFalse((target&control_mask).any())
        for a,b,original in zip(controls,again,regions):
            self.assertEqual((a['x'],a['y']),(b['x'],b['y']))
            np.testing.assert_equal(a['mask'],original['mask'])

    def test_unplaceable_controls_fail_instead_of_reducing_budget(self):
        domain=np.ones((4,4),dtype=bool)
        regions=[dict(x=0,y=0,mask=domain.copy())]
        with self.assertRaises(ValueError):
            random_controls(regions,domain,domain,domain,1,attempts=5)

    def fixture(self,root):
        profile=dict(driver='GTiff',width=32,height=32,count=1,dtype='uint16',
                     crs='EPSG:32648',transform=from_origin(100,200,1,1))
        for name in ('image','prediction','coverage','mask','gt'):
            cfg=profile.copy()
            data=np.ones((1,32,32),dtype=np.uint16)
            if name=='image':
                cfg.update(count=3,dtype='uint8')
                data=np.full((3,32,32),100,dtype=np.uint8)
            with rasterio.open(root/f'{name}.tif','w',**cfg) as dst:
                dst.write(data)
        (root/'weights.pth').write_bytes(b'synthetic')
        args=SimpleNamespace(image=root/'image.tif',prediction=root/'prediction.tif',
                             coverage=root/'coverage.tif',eval_mask=root/'mask.tif',checkpoint=root/'weights.pth',
                             output_dir=root/'tiles',gt=None,tile_size=16,stride=8,context=0,max_windows=0,
                             fusion='center',model_type='vit_b',device='cpu',rgb_bands=[1,2,3],rgb_range=None,
                             allow_crs_mismatch=False,boundary_tolerance=0)
        def initial(rgb,original,predictor,options):
            result=original.copy()
            result[3,3]=0
            return result,[]
        run_tiles(args,refine=initial)
        refusion(root/'tiles',root/'fusion',methods=['union'])
        return SimpleNamespace(union_dir=root/'fusion'/'union',checkpoint=root/'weights.pth',
                               output_dir=root/'second',gt=root/'gt.tif',device='cpu',
                               core_size=8,context=2,write_buffer=0,min_conflict=1,
                               max_regions=2,seed=41,plan_only=False)

    def test_second_stage_only_writes_planned_pixels_and_gt_is_opened_last(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=self.fixture(root)
            snapshot={p:p.read_bytes() for parent in (root/'tiles',root/'fusion')
                      for p in parent.rglob('*') if p.is_file()}
            calls=[]
            def refine(rgb,original,predictor,options):
                calls.append(1)
                result=original.copy()
                result[2,2]=0
                return result,[dict(source_id=1,component=1,accepted_count=1,prompt_count=1)]
            actual_open=rasterio.open
            def observed_open(path,*a,**kw):
                if Path(path).resolve()==args.gt.resolve():
                    self.assertEqual(len(calls),4)
                return actual_open(path,*a,**kw)
            with patch('rasterio.open',side_effect=observed_open):
                report=run_conflicts(args,refine=refine)
            self.assertEqual(report['status'],'complete')
            self.assertTrue((args.output_dir/'comparison.csv').exists())
            self.assertEqual(report['processed_regions'],2)
            from experiments.constrain_regions import run as replay
            second_snapshot={p:p.read_bytes() for p in args.output_dir.rglob('*') if p.is_file()}
            def replay_open(path,*a,**kw):
                if Path(path).resolve()==args.gt.resolve():
                    self.assertTrue((root/'bounded'/'conflict'/'instance.tif').exists())
                    self.assertTrue((root/'bounded'/'random'/'instance.tif').exists())
                return actual_open(path,*a,**kw)
            with patch('rasterio.open',side_effect=replay_open):
                bounded=replay(args.output_dir,root/'bounded',args.gt,radius=1)
            self.assertEqual(bounded['status'],'complete')
            self.assertEqual(len(bounded['evaluation']),6)
            from experiments.report_regions import build
            build(root/'bounded',args.gt,root/'visual_report',size=8,per_kind=1)
            self.assertTrue((root/'visual_report'/'index.html').exists())
            self.assertTrue((root/'visual_report'/'cases.json').exists())
            with self.assertRaises(FileExistsError):
                build(root/'bounded',args.gt,root/'visual_report')
            for method in ('conflict','random'):
                old=bounded['evaluation'][method+'_unrestricted']['transitions_from_union']
                new=bounded['evaluation'][method+'_boundary']['transitions_from_union']
                for key in ('removed_true_foreground','removed_false_foreground','added_true_foreground','added_false_foreground'):
                    self.assertLessEqual(new[key],old[key])
                self.assertLessEqual(bounded['results'][method]['retained_changed_pixels'],
                                     bounded['results'][method]['unrestricted_changed_pixels'])
            for path,data in second_snapshot.items():
                self.assertEqual(path.read_bytes(),data)
            with self.assertRaises(FileExistsError):
                replay(args.output_dir,root/'bounded',args.gt)
            from experiments.diagnose_regions import diagnose
            diagnostic=diagnose(args.output_dir,args.gt,root/'diagnosis')
            for method in ('conflict','random'):
                rows=[r for r in diagnostic['rows'] if r['method']==method and r['location']=='all']
                total=next(r for r in rows if r['region']=='all_roi')
                parts=[r for r in rows if r['region'] in ('selected_raw_conflict','write_outside_raw_conflict','outside_write')]
                for key in ('removed_true_foreground','removed_false_foreground','added_true_foreground','added_false_foreground'):
                    self.assertEqual(total[key],sum(r[key] for r in parts))
            self.assertTrue((root/'diagnosis'/'summary.txt').exists())
            with self.assertRaises(FileExistsError):
                diagnose(args.output_dir,args.gt,root/'diagnosis')
            with rasterio.open(args.union_dir/'instance.tif') as src:
                union=src.read(1)
            for method in ('conflict','random'):
                with rasterio.open(args.output_dir/f'{method}_write.tif') as src:
                    allowed=src.read(1)>0
                with rasterio.open(args.output_dir/method/'instance.tif') as src:
                    result=src.read(1)
                self.assertFalse(np.any((result!=union)&~allowed))
                self.assertEqual(report['results'][method]['prompt_calls'],2)
            for path,data in snapshot.items():
                self.assertEqual(path.read_bytes(),data)

    def test_no_accepted_candidate_keeps_union_and_plan_only_skips_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=self.fixture(root)
            def fallback(rgb,original,predictor,options):
                return original.copy(),[dict(source_id=1,component=1,accepted_count=0,prompt_count=1)]
            report=run_conflicts(args,refine=fallback)
            self.assertEqual(report['results']['conflict']['changed_pixels'],0)
            self.assertEqual(report['results']['random']['changed_pixels'],0)
            args.output_dir=root/'plan'
            args.plan_only=True
            args.gt=root/'does-not-exist.tif'
            report=run_conflicts(args,refine=lambda *a: self.fail('Model called during planning'))
            self.assertEqual(report['status'],'planned')
            self.assertFalse((args.output_dir/'comparison.csv').exists())


if __name__=='__main__':
    unittest.main()
