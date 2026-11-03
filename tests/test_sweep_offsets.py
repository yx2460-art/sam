import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio

import test_sweep_fusions
import test_tiles
from experiments.run_tiles import run
from experiments.sweep_fusions import sweep as fusion_sweep
from experiments.sweep_offsets import phases, sweep
from experiments.tiles import windows


class OffsetTests(unittest.TestCase):
    def test_shifted_grid_covers_edges_and_preserves_pixel_coordinates(self):
        for size,stride in ((4,4),(8,4),(8,2)):
            for _,dx,dy in phases(stride):
                coverage=np.zeros((11,13),dtype=np.uint32)
                seen=set()
                for w in windows(13,11,size,stride,(0,0,13,11),dx,dy):
                    key=tuple(w.flatten())
                    self.assertNotIn(key,seen);seen.add(key)
                    x,y,width,height=map(int,key)
                    self.assertTrue(0<=x<x+width<=13)
                    self.assertTrue(0<=y<y+height<=11)
                    if x>0: self.assertEqual(x%stride,dx)
                    if y>0: self.assertEqual(y%stride,dy)
                    coverage[y:y+height,x:x+width]+=1
                self.assertTrue((coverage>0).all())
        with self.assertRaises(ValueError):
            list(windows(13,11,8,4,(0,0,13,11),4,0))

    def test_full_offset_sweep_reuses_originals_and_reads_gt_after_all_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture=test_sweep_fusions.FusionSweepTests()
            dense=fixture.fixture(root)
            fusion_sweep(dense,root/'fusion')
            snapshot={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            calls=[]
            def execute(parameters):
                calls.append((parameters['stride'],parameters['offset_x'],parameters['offset_y']))
                options=dict(parameters,max_windows=0,gt=None)
                for key in ('image','prediction','coverage','eval_mask','checkpoint','output_dir'):
                    options[key]=Path(options[key])
                run(SimpleNamespace(**options),refine=test_tiles.TileTests.identity)
            with patch('experiments.sweep_offsets.TARGETS',((8,4,'union'),(8,2,'majority'))):
                plan=sweep(root/'fusion',root/'offsets',True)
                self.assertEqual(sum(j['status']=='planned' for j in plan['jobs']),6)
                actual_open=rasterio.open
                def observed(path,*a,**kw):
                    if Path(path).resolve()==(root/'gt.tif').resolve():
                        self.assertEqual(len(calls),6)
                    return actual_open(path,*a,**kw)
                with patch('rasterio.open',side_effect=observed):
                    result=sweep(root/'fusion',root/'offsets',runner=execute)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(len(result['rows']),8)
                self.assertTrue((root/'offsets'/'cases.json').exists())
                self.assertTrue((root/'offsets'/'index.html').exists())
                again=sweep(root/'fusion',root/'offsets',runner=lambda p:self.fail('Unexpected inference'))
                self.assertEqual(again['rows'],result['rows'])
                for path,data in snapshot.items():
                    self.assertEqual(path.read_bytes(),data)
                from experiments.sweep_strides import compatible,read_json
                shifted=next(j for j in result['jobs'] if j['phase']=='both')
                m=read_json(Path(shifted['run_dir'])/'manifest.json')
                signature=read_json(dense/'sweep.json')['signature']
                self.assertFalse(compatible(m,signature['parameters'],signature['hashes'],8,4,signature['prompt']))
