import json
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
            scored=evaluate(root/'chosen',args.gt,root/'evaluation')
            self.assertEqual(set(scored['evaluation']),{'center','union','intersection','manual'})
            self.assertTrue((root/'evaluation'/'summary.txt').is_file())
            self.assertTrue((project/'regions.geojson').is_file())
            self.assertNotIn('GT boundaries',(project/'index.html').read_text(encoding='utf-8'))
            for p,data in snapshot.items(): self.assertEqual(p.read_bytes(),data)
            decisions['decisions'][0]['reviewed']=False
            path.write_text(json.dumps(decisions))
            with self.assertRaises(ValueError): apply(project,path,root/'invalid')
            decisions['decisions'][0]['reviewed']=True
            decisions['decisions'].append(decisions['decisions'][0]);path.write_text(json.dumps(decisions))
            with self.assertRaises(ValueError): apply(project,path,root/'duplicate')
            with (project/'write_domain.tif').open('ab') as stream: stream.write(b'changed')
            with self.assertRaises(ValueError): apply(project,path,root/'tampered')


if __name__=='__main__': unittest.main()
