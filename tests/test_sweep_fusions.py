import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_tiles
from experiments.run_tiles import run
from experiments.sweep_strides import sweep as stride_sweep
from experiments.sweep_fusions import plot, sweep


class FusionSweepTests(unittest.TestCase):
    def fixture(self,root):
        fixture=test_tiles.TileTests()
        args,_=fixture.fixture(root)
        args.gt=None
        def shrink(rgb,original,predictor,options):
            result=original.copy()
            result[:2]=0
            return result,[]
        run(args,refine=shrink)
        def execute(parameters):
            options=dict(parameters,max_windows=0,gt=None)
            for key in ('image','prediction','coverage','eval_mask','checkpoint','output_dir'):
                options[key]=Path(options[key])
            run(SimpleNamespace(**options),refine=shrink)
        with patch('experiments.sweep_strides.CONFIGURATIONS',[(8,4),(8,2)]):
            stride_sweep(args.output_dir,root/'gt.tif',root/'dense',runner=execute)
        return root/'dense'

    def test_cached_rules_resume_reproduce_center_and_preserve_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            dense=self.fixture(root)
            sources={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            plan=sweep(dense,root/'fusion',True)
            self.assertEqual([j['status'] for j in plan['jobs']],['planned','planned'])
            result=sweep(dense,root/'fusion')
            self.assertEqual(result['status'],'complete')
            self.assertFalse(result['sam_inference_performed'])
            self.assertEqual(len(result['rows']),8)
            centers=[r for r in result['rows'] if r['method']=='center']
            self.assertTrue(all(r['delta_mask_iou_vs_center']==0 for r in centers))
            self.assertTrue((root/'fusion'/'index.html').exists())
            with patch('experiments.sweep_fusions.refusion',side_effect=AssertionError('Repeated fusion')):
                again=sweep(dense,root/'fusion')
            self.assertEqual(again['rows'],result['rows'])
            for path,data in sources.items():
                self.assertEqual(path.read_bytes(),data)
            cache=Path(result['jobs'][0]['run_dir'])/'work'/'votes.bin'
            original=cache.read_bytes()
            changed=bytearray(original)
            changed[0]^=1
            cache.write_bytes(changed)
            with self.assertRaises(ValueError):
                sweep(dense,root/'fusion')
            cache.write_bytes(original)

    def test_missing_cache_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            dense=self.fixture(root)
            from experiments.sweep_strides import read_json
            job=read_json(dense/'sweep.json')['jobs'][0]
            cache=Path(job['run_dir'])/'work'/'votes.bin'
            cache.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                sweep(dense,root/'fusion',True)
            self.assertFalse((root/'fusion').exists())

    def test_plot_shows_all_four_rules(self):
        rows=[dict(tile_size=256,stride=128,overlap_fraction=.5,method=m,mask_iou=.85)
              for m in ('center','union','intersection','majority')]
        svg=plot(rows,256,'mask_iou',.86)
        root=ET.fromstring(svg)
        self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}circle')),4)
        for name in ('center','union','intersection','majority','BsiNet'):
            self.assertIn(name,svg)
