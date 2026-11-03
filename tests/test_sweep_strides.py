import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rasterio

from experiments.run_tiles import run
from experiments.sweep_strides import CONFIGURATIONS, curve, sweep
import test_tiles


class DenseSweepTests(unittest.TestCase):
    def test_default_matrix_has_seven_strides_per_size(self):
        self.assertEqual(CONFIGURATIONS,[(s,s*n//8) for s in (256,512) for n in (8,7,6,5,4,3,2)])
        self.assertEqual(len(set(CONFIGURATIONS)),14)

    def test_curves_keep_actual_overlap_positions_and_baseline(self):
        rows=[dict(tile_size=size,stride=stride,overlap_fraction=1-stride/size,mask_iou=.8+stride/size*.02)
              for size,stride in CONFIGURATIONS]
        svg=curve(rows,'mask_iou','IoU',.83)
        root=ET.fromstring(svg)
        self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}circle')),14)
        self.assertIn('BsiNet baseline',svg)
        self.assertIn('75%',svg)

    def test_plan_reuse_run_resume_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture=test_tiles.TileTests()
            args,_=fixture.fixture(root)
            args.gt=None
            run(args,refine=fixture.identity)
            snapshot={p:p.read_bytes() for p in args.output_dir.rglob('*') if p.is_file()}
            calls=[]
            def execute(p):
                calls.append((p['tile_size'],p['stride']))
                opts=dict(p,max_windows=0,gt=None)
                for key in ('image','prediction','coverage','eval_mask','checkpoint','output_dir'):
                    opts[key]=Path(opts[key])
                run(SimpleNamespace(**opts),refine=fixture.identity)
            with patch('experiments.sweep_strides.CONFIGURATIONS',[(8,4),(8,2)]):
                planned=sweep(args.output_dir,root/'gt.tif',root/'sweep',True)
                self.assertEqual([j['status'] for j in planned['jobs']],['complete','planned'])
                self.assertEqual(calls,[])
                actual_open=rasterio.open
                def observed(path,*a,**kw):
                    if Path(path).resolve()==(root/'gt.tif').resolve():
                        self.assertEqual(calls,[(8,2)])
                    return actual_open(path,*a,**kw)
                with patch('rasterio.open',side_effect=observed):
                    completed=sweep(args.output_dir,root/'gt.tif',root/'sweep',runner=execute)
                self.assertEqual(completed['status'],'complete')
                self.assertEqual(len(completed['rows']),2)
                self.assertTrue((root/'sweep'/'index.html').exists())
                self.assertTrue((root/'sweep'/'mask_iou.svg').exists())
                self.assertTrue((root/'sweep'/'summary.txt').exists())
                again=sweep(args.output_dir,root/'gt.tif',root/'sweep',runner=lambda p:self.fail('Unexpected repeat inference'))
                self.assertEqual(again['rows'],completed['rows'])
                self.assertEqual(completed['rows'][0]['mask_iou'],1)
                self.assertEqual(completed['rows'][1]['mask_iou'],1)
                for path,data in snapshot.items():
                    self.assertEqual(path.read_bytes(),data)
                # A changed input must invalidate resume rather than mix experiments.
                (root/'weights.pth').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    sweep(args.output_dir,root/'gt.tif',root/'sweep',runner=execute)

    def test_partial_window_run_is_not_silently_restarted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture=test_tiles.TileTests()
            args,_=fixture.fixture(root)
            args.gt=None
            run(args,refine=fixture.identity)
            with patch('experiments.sweep_strides.CONFIGURATIONS',[(8,2)]):
                sweep(args.output_dir,root/'gt.tif',root/'sweep',True)
                directory=root/'sweep'/'runs'/'t8_s2'
                directory.mkdir(parents=True)
                (directory/'manifest.json').write_text(json.dumps({'status':'running'}))
                with self.assertRaises(ValueError):
                    sweep(args.output_dir,root/'gt.tif',root/'sweep')
