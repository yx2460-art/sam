import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio

import test_conflict_regions
from experiments.manual_fusion import prepare,apply,evaluate
from experiments.run_tiles import digest


class ManualFusionTests(unittest.TestCase):
    def test_selection_is_gt_free_scoring_is_separate_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=test_conflict_regions.RegionTests().fixture(root)
            snapshot={p:p.read_bytes() for p in (root/'tiles').rglob('*') if p.is_file()}
            actual_open=rasterio.open
            def no_gt(path,*a,**kw):
                self.assertNotEqual(Path(path).resolve(),args.gt.resolve())
                return actual_open(path,*a,**kw)
            project=root/'manual'
            with patch('rasterio.open',side_effect=no_gt):
                m=prepare(root/'tiles',project,size=8)
                decisions=dict(project_sha256=digest(project/'project.json'),decisions=[
                    dict(id=r['id'],choice=('intersection' if r['id']==0 else 'keep'),reviewed=r['id']==0)
                    for r in m['regions']])
                path=root/'decisions.json';path.write_text(json.dumps(decisions))
                report=apply(project,path,root/'chosen')
            with rasterio.open(m['candidates']['center']) as src: initial=src.read(1)
            with rasterio.open(root/'chosen'/'instance.tif') as src: result=src.read(1)
            np.testing.assert_equal(result[8:],initial[8:])
            np.testing.assert_equal(result[:8,8:],initial[:8,8:])
            with rasterio.open(m['candidates']['intersection']) as src:
                np.testing.assert_equal(result[:8,:8],src.read(1)[:8,:8])
            self.assertEqual(report['selected_regions'],1)
            with rasterio.open(root/'chosen'/'changed_pixels.tif') as src:
                np.testing.assert_equal(src.read(1)>0,result!=initial)
            with rasterio.open(root/'chosen'/'selected_write.tif') as src:
                self.assertEqual(int(src.read(1).sum()),64)
            self.assertTrue((root/'chosen'/'regions.csv').is_file())
            self.assertIn('GT not read',(root/'chosen'/'summary.txt').read_text())
            scored=evaluate(root/'chosen',args.gt,root/'evaluation')
            self.assertEqual(set(scored['evaluation']),{'center','union','intersection','manual'})
            self.assertTrue((root/'evaluation'/'summary.txt').is_file())
            self.assertTrue((project/'regions.geojson').is_file())
            self.assertNotIn('GT boundaries',(project/'index.html').read_text(encoding='utf-8'))
            if shutil.which('node'):
                script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(0,'utf8'),script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map();let exported;const c={Date,JSON,Object,Set,Number,Error,setTimeout:fn=>fn(),
Blob:class{constructor(parts){exported=JSON.parse(parts.join(''));}},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},location:{},
localStorage:{getItem:()=>null,setItem:()=>{}},document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{value:'keep'});return elements.get(id);},createElement:()=>({click(){}})}};
vm.createContext(c);vm.runInContext(script,c);
(async()=>{elements.get('choice0').value='union';c.record(0);c.download();
assert.equal(elements.get('preview0').src,'region_0_union.png');
const saved=JSON.parse(JSON.stringify(exported));elements.get('choice0').value='keep';c.record(0);
await c.restore({text:async()=>JSON.stringify(saved)});assert.equal(elements.get('choice0').value,'union');
assert(elements.get('progress').textContent.includes('1/16'));
saved.project_sha256='wrong';await c.restore({text:async()=>JSON.stringify(saved)});
assert(elements.get('message').textContent.includes('导入失败'));assert.equal(elements.get('choice0').value,'union');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
                checked=subprocess.run(['node','-e',script],input=(project/'index.html').read_text(encoding='utf-8'),
                                       text=True,encoding='utf-8',capture_output=True)
                self.assertEqual(checked.returncode,0,checked.stderr)
            for p,data in snapshot.items(): self.assertEqual(p.read_bytes(),data)
            decisions['decisions'][0]['reviewed']=False
            path.write_text(json.dumps(decisions))
            with self.assertRaises(ValueError): apply(project,path,root/'invalid')
            decisions['decisions'][0]['reviewed']=True
            decisions['session_wall_seconds']=[-1];path.write_text(json.dumps(decisions))
            with self.assertRaises(ValueError): apply(project,path,root/'bad_time')
            decisions['session_wall_seconds']=[]
            decisions['decisions'].append(decisions['decisions'][0]);path.write_text(json.dumps(decisions))
            with self.assertRaises(ValueError): apply(project,path,root/'duplicate')
            with (project/'write_domain.tif').open('ab') as stream: stream.write(b'changed')
            with self.assertRaises(ValueError): apply(project,path,root/'tampered')


if __name__=='__main__': unittest.main()
