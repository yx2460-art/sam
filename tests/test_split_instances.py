import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments.run_tiles import digest
from experiments.split_instances import prepare,infer,apply,evaluate,split_candidate


class Predictor:
    model=SimpleNamespace(mask_threshold=0)
    def __init__(self,ambiguous=False): self.ambiguous=ambiguous
    def set_image(self,rgb): self.shape=rgb.shape[:2]
    def predict(self,point_coords,point_labels,box,**kwargs):
        assert list(point_labels)==[1]
        if box is None: box=[0,0,self.shape[1]-1,self.shape[0]-1]
        raw=np.full(self.shape,-5,dtype=float)
        yy,xx=np.indices(self.shape);middle=(box[1]+box[3])/2
        mask=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
        if not self.ambiguous:
            mask &= (yy<=middle) if point_coords[0,1]<=middle else (yy>middle)
        raw[mask]=5
        return np.array([raw]),np.array([.95]),None


class SplitTests(unittest.TestCase):
    def test_point_only_removes_box_and_preserves_positive_labels_and_thresholds(self):
        class Spy(Predictor):
            def __init__(self): super().__init__();self.boxes=[]
            def predict(self,point_coords,point_labels,box,**kwargs):
                self.boxes.append(box)
                return super().predict(point_coords,point_labels,box,**kwargs)
        parent=np.ones((20,20),dtype=bool);rgb=np.zeros((20,20,3),dtype=np.uint8)
        for mode in ('point-box','point-only'):
            predictor=Spy()
            result,details=split_candidate(rgb,parent,[[5,3],[5,16]],predictor,min_area=3,
                                           source=parent.astype(np.uint32),prompt_mode=mode)
            self.assertIsNotNone(result)
            self.assertEqual(len(predictor.boxes),2)
            self.assertTrue(all((b is None)==(mode=='point-only') for b in predictor.boxes))
            self.assertTrue(all(d['prompt_mode']==mode for d in details['diagnostics']))
            self.assertEqual(details['diagnostics'][0]['candidates'][0]['min_score'],.8)

    def fixture(self,root):
        labels=np.zeros((24,24),dtype=np.uint32);labels[3:21,3:12]=1;labels[3:21,12:21]=2
        truth=labels.copy();truth[3:12,3:12]=10;truth[12:21,3:12]=11
        truth[3:12,12:21]=12;truth[12:21,12:21]=13
        profile=dict(driver='GTiff',width=24,height=24,count=1,dtype='uint32',transform=from_origin(100,200,1,1),crs='EPSG:32648')
        for name,data in [('instances',labels),('gt',truth),('roi',(labels>0).astype(np.uint32))]:
            with rasterio.open(root/f'{name}.tif','w',**profile) as dst:dst.write(data,1)
        with rasterio.open(root/'image.tif','w',**dict(profile,count=3,dtype='uint8')) as dst:
            dst.write(np.full((3,24,24),100,dtype=np.uint8))
        (root/'weights.pth').write_bytes(b'synthetic')
        return labels

    def test_positive_only_rejects_indistinguishable_masks_and_bad_points(self):
        mask=np.ones((20,20),dtype=bool);rgb=np.zeros((20,20,3),dtype=np.uint8)
        proposal,detail=split_candidate(rgb,mask,[[5,3],[5,16]],Predictor(True),min_area=3)
        self.assertIsNone(proposal)
        self.assertIn('cannot separate',detail['reason'])
        for points in ([[5,3],[5,3]],[[30,3],[5,16]],[[5.5,3],[5,16]]):
            with self.assertRaises(ValueError):split_candidate(rgb,mask,points,Predictor(),3)

    def test_diagnostics_report_all_rules_without_changing_candidate_selection(self):
        class Mixed(Predictor):
            def predict(self,point_coords,**kwargs):
                x,y=map(int,point_coords[0]);shape=self.shape
                whole=np.full(shape,5.);small=np.full(shape,-5.);small[y,x]=5
                missing=np.full(shape,-5.);good=np.full(shape,-5.)
                good[:10 if y<10 else 0]=5
                if y>=10: good[10:]=5
                bad=good.copy();bad[0,0]=np.nan
                return np.array([whole,small,missing,good,bad,good]),np.array([.6,.95,.99,.92,.95,np.nan]),None
        rgb=np.zeros((20,20,3),dtype=np.uint8);parent=np.ones((20,20),dtype=bool)
        labels,detail=split_candidate(rgb,parent,[[4,4],[15,15]],Mixed(),min_area=3,source=parent.astype(np.uint32))
        self.assertEqual(set(np.unique(labels)),{1,2})
        for seed in detail['diagnostics']:
            candidates=seed['candidates']
            self.assertEqual(set(candidates[0]['rejection_codes']),{'low_score','contains_other_points'})
            self.assertIn('too_small',candidates[1]['rejection_codes'])
            self.assertIn('own_point_missing',candidates[2]['rejection_codes'])
            self.assertTrue(candidates[3]['selected'])
            self.assertEqual(candidates[3]['rejection_codes'],[])
            self.assertIn('non_finite_mask',candidates[4]['rejection_codes'])
            self.assertIn('non_finite_score',candidates[5]['rejection_codes'])
            self.assertIsNone(candidates[5]['score'])
            self.assertEqual(seed['status'],'retained')

    def test_full_instance_new_ids_approval_and_overlapping_contexts_preserve_other_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=self.fixture(root);snapshot=(root/'instances.tif').read_bytes()
            actual=rasterio.open
            def no_gt(path,*args,**kwargs):
                self.assertNotEqual(Path(path).resolve(),(root/'gt.tif').resolve())
                return actual(path,*args,**kwargs)
            with patch('rasterio.open',side_effect=no_gt):
                m=prepare(root/'image.tif',root/'instances.tif',root/'project',roi_path=root/'roi.tif',pad=2)
                self.assertEqual([r['source_pixels'] for r in m['regions']],[162,162])
                points=[]
                for row in m['regions']:
                    x0,y0,_,_=row['xyxy'];x=6 if row['id']==1 else 17
                    points.append(dict(id=row['id'],points=[[x-x0,6-y0],[x-x0,17-y0]]))
                (root/'points.json').write_text(json.dumps(dict(project_sha256=digest(root/'project'/'project.json'),instances=points)))
                proposals=infer(root/'project',root/'points.json',root/'weights.pth',root/'proposals',min_area=4,predictor=Predictor())
                self.assertTrue(all(p['status']=='proposal' for p in proposals['proposals']))
                self.assertTrue((root/'proposals'/'diagnostics.html').is_file())
                self.assertTrue((root/'proposals'/'diagnostics.csv').is_file())
                diagnostics=json.loads((root/'proposals'/'diagnostics.json').read_text(encoding='utf-8'))
                self.assertFalse(diagnostics['ground_truth_used'])
                record=diagnostics['instances'][0]['diagnostics'][0]['candidates'][0]
                with np.load(root/'proposals'/record['mask_archive']) as archive:
                    self.assertEqual(set(archive.files),{'raw','supported','component'})
                    self.assertEqual(archive['raw'].dtype,np.bool_)
                self.assertTrue((root/'proposals'/record['images']['raw']).is_file())
                if shutil.which('node'):
                    check=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');const pages=JSON.parse(fs.readFileSync(0,'utf8'));
function harness(page){const nodes={},canvases=[...page.matchAll(/<canvas id="([^"]+)" width="(\d+)" height="(\d+)"/g)];
canvases.forEach(m=>nodes[m[1]]={width:+m[2],height:+m[3],getBoundingClientRect:()=>({left:0,top:0,width:+m[2]*10,height:+m[3]*10}),getContext:()=>({drawImage(){},beginPath(){},arc(){},fill(){},fillText(){}})});
let payload;const c={Date,JSON,Object,Math,Image:class{set src(v){this.onload();}},alert:()=>{throw Error('Unexpected alert');},
Blob:class{constructor(parts){payload=JSON.parse(parts.join(''));}},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},setTimeout:fn=>fn(),
document:{getElementById:id=>nodes[id]||(nodes[id]={checked:false}),createElement:()=>({click(){}})}};
vm.createContext(c);vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],c);return {c,nodes,get:()=>payload};}
const points=harness(pages[0]);points.nodes.c1.onclick({clientX:55,clientY:55});points.nodes.c1.onclick({clientX:55,clientY:165});points.c.download();
assert.deepEqual(points.get().instances,[{id:1,points:[[5,5],[5,16]]}]);
points.nodes.c1.onclick({clientX:15,clientY:15});assert(points.nodes.s1.textContent.includes('预测背景'));points.c.undo(1);
points.nodes.c1.onclick({clientX:115,clientY:55});assert(points.nodes.s1.textContent.includes('已有实例ID 2'));points.c.undo(1);
points.c.undo(1);assert(points.nodes.s1.textContent.startsWith('1 '));
const review=harness(pages[1]);review.c.download();assert.deepEqual(review.get().accepted_ids,[]);
review.nodes.a1.checked=true;review.c.download();assert.deepEqual(review.get().accepted_ids,[1]);
'''
                    checked=subprocess.run(['node','-e',check],input=json.dumps([
                        (root/'project'/'index.html').read_text(encoding='utf-8'),
                        (root/'proposals'/'review.html').read_text(encoding='utf-8')]),
                        text=True,encoding='utf-8',capture_output=True)
                    self.assertEqual(checked.returncode,0,checked.stderr)
                approve=dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[1,2])
                (root/'approvals.json').write_text(json.dumps(approve))
                result=apply(root/'proposals',root/'approvals.json',root/'result')
                with actual(root/'result'/'instance.tif') as src:
                    output=src.read(1);self.assertEqual(src.shape,original.shape)
                np.testing.assert_equal(output>0,original>0)
                self.assertEqual(set(np.unique(output)),{0,3,4,5,6})
                self.assertEqual(len(result['accepted']),2)
                approve['accepted_ids']=[1];(root/'approvals.json').write_text(json.dumps(approve))
                apply(root/'proposals',root/'approvals.json',root/'one')
                with actual(root/'one'/'instance.tif') as src:one=src.read(1)
                np.testing.assert_equal(one[original==2],original[original==2])
                approve['accepted_ids']=[];(root/'approvals.json').write_text(json.dumps(approve))
                apply(root/'proposals',root/'approvals.json',root/'cancel')
                with actual(root/'cancel'/'instance.tif') as src:np.testing.assert_equal(src.read(1),original)
            metrics=evaluate(root/'result',root/'gt.tif',root/'roi.tif',root/'eval',tolerance=0)
            self.assertEqual(metrics['evaluation']['after']['mask_iou'],1)
            self.assertGreater(metrics['evaluation']['after']['instance_f1'],metrics['evaluation']['before']['instance_f1'])
            self.assertEqual(snapshot,(root/'instances.tif').read_bytes())
            with (root/'proposals'/'1_proposal.tif').open('ab') as stream:stream.write(b'tampered')
            with self.assertRaises(ValueError):apply(root/'proposals',root/'approvals.json',root/'invalid')

    def test_large_context_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            with self.assertRaises(ValueError):prepare(root/'image.tif',root/'instances.tif',root/'project',ids=[1],pad=2,max_side=5)
            self.assertFalse((root/'project').exists())

    def test_background_positive_adds_foreground_other_instance_point_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=self.fixture(root)
            # Use the old project format/hash flow: existing exported points must remain usable.
            m=prepare(root/'image.tif',root/'instances.tif',root/'project',ids=[1],pad=4)
            x0,y0,_,_=m['regions'][0]['xyxy']
            points=[[1-x0,1-y0],[6-x0,17-y0],[14-x0,17-y0]]
            annotation=dict(project_sha256=digest(root/'project'/'project.json'),instances=[dict(id=1,points=points)])
            (root/'points.json').write_text(json.dumps(annotation))
            actual=rasterio.open
            def no_gt(path,*a,**kw):
                self.assertNotEqual(Path(path).resolve(),(root/'gt.tif').resolve())
                return actual(path,*a,**kw)
            with patch('rasterio.open',side_effect=no_gt):
                proposals=infer(root/'project',root/'points.json',root/'weights.pth',root/'proposals',min_area=4,predictor=Predictor())
                p=proposals['proposals'][0]
                self.assertEqual([r['kind'] for r in p['point_roles']],['background','parent','other_instance'])
                self.assertEqual(p['status'],'proposal')
                self.assertGreater(p['detail']['added_foreground_pixels'],0)
                self.assertTrue(any(r['seed']==2 for r in p['detail']['rejected_seeds']))
                self.assertEqual(p['detail']['diagnostics'][2]['status'],'protected')
                self.assertEqual(p['detail']['diagnostics'][2]['candidates'],[])
                approval=dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[1])
                (root/'approval.json').write_text(json.dumps(approval))
                apply(root/'proposals',root/'approval.json',root/'result')
            with actual(root/'result'/'instance.tif') as src:edited=src.read(1)
            self.assertGreater(int((edited>0).sum()),int((original>0).sum()))
            np.testing.assert_equal(edited[original==2],original[original==2])
            self.assertTrue(np.all(edited[original>0]>0))
            self.assertNotIn('点击其他实例无效',(root/'project'/'index.html').read_text(encoding='utf-8'))

    def test_one_background_point_allowed_and_no_candidate_is_not_a_validation_error(self):
        original=np.zeros((20,20),dtype=np.uint32);original[10:18,2:8]=1
        parent=original==1;rgb=np.zeros((20,20,3),dtype=np.uint8)
        assigned,details=split_candidate(rgb,parent,[[3,3]],Predictor(),3,source=original)
        self.assertIsNotNone(assigned)
        self.assertEqual(details['children'][0]['kind'],'background_completion')
        original[1,1]=2
        assigned,details=split_candidate(rgb,parent,[[1,1]],Predictor(),3,source=original)
        self.assertIsNone(assigned)
        self.assertEqual(details['rejected_seeds'][0]['source_id'],2)

    def test_overlapping_background_additions_are_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            m=prepare(root/'image.tif',root/'instances.tif',root/'project',ids=[1,2],pad=24)
            points=dict(project_sha256=digest(root/'project'/'project.json'),instances=[
                dict(id=r['id'],points=[[1,1]]) for r in m['regions']])
            (root/'points.json').write_text(json.dumps(points))
            proposals=infer(root/'project',root/'points.json',root/'weights.pth',root/'proposals',min_area=4,predictor=Predictor())
            self.assertTrue(all(p['status']=='proposal' for p in proposals['proposals']))
            approval=dict(proposals_sha256=digest(root/'proposals'/'proposals.json'),accepted_ids=[1,2])
            (root/'approval.json').write_text(json.dumps(approval))
            with self.assertRaisesRegex(ValueError,'overlap'):
                apply(root/'proposals',root/'approval.json',root/'result')
            self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main()
