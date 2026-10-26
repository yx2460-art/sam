import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments.check_full import audit, footprint, load_split


class FullCheckTests(unittest.TestCase):
    def setup_rasters(self, root, mismatch=False):
        profile = dict(driver='GTiff', width=8, height=8, count=1, dtype='uint8',
                       crs='EPSG:32648', transform=from_origin(100, 200, 1, 1))
        for name in ('image', 'gt', 'prediction', 'coverage'):
            data = np.zeros((1, 8, 8), dtype=np.uint8)
            current = profile.copy()
            if name == 'image':
                data = np.full((3, 8, 8), 100, dtype=np.uint8)
                current['count'] = 3
            elif name == 'coverage':
                data[:, :, :4] = 1
            else:
                data[:, 2:4, 2:6] = 1
                current['nodata'] = 0
            if mismatch and name == 'prediction':
                current['transform'] = from_origin(101, 200, 1, 1)
            with rasterio.open(root / f'{name}.tif', 'w', **current) as dst:
                dst.write(data)
        return SimpleNamespace(**{key:root / f'{key}.tif' for key in (
            'image', 'gt', 'prediction', 'coverage')}, output_dir=root / 'check',
            annotation_roi=None, split_dir=None, tile_size=4, train_buffer=0)

    def test_counts_zero_nodata_and_no_implicit_evaluation_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = audit(self.setup_rasters(root))
            self.assertEqual(result['counts']['coverage'], 32)
            self.assertEqual(result['counts']['gt_outside_coverage'], 4)
            self.assertEqual(result['counts']['image_valid'], 64)
            self.assertFalse(result['evaluation_ready'])
            self.assertTrue((root / 'check' / 'preview.tif').exists())
            self.assertTrue((root / 'check' / 'report.txt').exists())

    def test_grid_mismatch_reports_failure_without_reading_misaligned_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.setup_rasters(Path(folder), mismatch=True)
            result = audit(args)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(any('affine' in error for error in result['errors']))
            self.assertFalse(result['counts'])
            with self.assertRaises(FileExistsError):
                audit(args)

    def test_crs_override_is_explicit_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.setup_rasters(root)
            with rasterio.open(args.prediction, 'r+') as dst:
                dst.crs = 'EPSG:32649'
            before = args.prediction.read_bytes()
            self.assertEqual(audit(args)['status'], 'failed')
            args.output_dir = root / 'override'
            args.allow_crs_mismatch = True
            result = audit(args)
            self.assertFalse(result['errors'])
            self.assertEqual(result['counts']['coverage'], 32)
            self.assertEqual(result['crs_mismatch_bypassed_for'], ['prediction'])
            self.assertFalse(result['evaluation_ready'])
            self.assertFalse(result['crs_equivalence_confirmed'])
            self.assertEqual(args.prediction.read_bytes(), before)

    def test_override_does_not_allow_grid_size_or_missing_crs(self):
        cases = [(8, 101, 'EPSG:32649'), (7, 100, 'EPSG:32649'), (8, 100, None)]
        for width, origin, crs in cases:
            with self.subTest(width=width, crs=crs), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                args = self.setup_rasters(root)
                with rasterio.open(args.prediction, 'w', driver='GTiff', width=width,
                                   height=8, count=1, dtype='uint8', crs=crs,
                                   transform=from_origin(origin, 200, 1, 1)) as dst:
                    dst.write(np.zeros((1, 8, width), dtype=np.uint8))
                args.allow_crs_mismatch = True
                result = audit(args)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['crs_mismatch_bypassed_for'], [])
                self.assertFalse(result['counts'])

    def test_split_overlap_is_excluded_and_missing_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.setup_rasters(root)
            (root / 'train_ids.txt').write_text('a\n')
            (root / 'val_ids.txt').write_text('b\n')
            with (root / 'tiles_index.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['filename', 'x', 'y'])
                writer.writeheader()
                writer.writerows([dict(filename='a.tif', x=0, y=0),
                                  dict(filename='b.tif', x=0, y=2)])
            args.split_dir = root
            result = audit(args)
            self.assertEqual(result['counts']['train_val_overlap'], 8)
            self.assertEqual(result['counts']['potential_validation'], 8)
            with rasterio.open(root / 'check' / 'potential_validation.tif') as src:
                array = src.read(1)
                self.assertEqual(int(array[:4].sum()), 0)
                self.assertEqual(int(array.sum()), 8)
            (root / 'val_ids.txt').write_text('missing\n')
            with self.assertRaises(ValueError):
                load_split(root, 4, 8, 8)

    def test_annotation_roi_limits_potential_mask_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.setup_rasters(root)
            before = {path: path.read_bytes() for path in root.glob('*.tif')}
            # Absent split remains unverified even if an annotation mask is supplied.
            args.annotation_roi = args.coverage
            result = audit(args)
            self.assertFalse(result['evaluation_ready'])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            saved = json.loads((args.output_dir / 'report.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['annotation_completeness_confirmed'])

    def test_multiple_splits_exclude_training_union_from_common_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.setup_rasters(root)
            directories = [root / 'one', root / 'two']
            for i, directory in enumerate(directories):
                directory.mkdir()
                (directory / 'train_ids.txt').write_text('a\n')
                (directory / 'val_ids.txt').write_text('b\n')
                with (directory / 'tiles_index.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['filename', 'x', 'y'])
                    writer.writeheader()
                    writer.writerows([dict(filename='a.tif', x=0, y=i*2),
                                      dict(filename='b.tif', x=0, y=4)])
            args.split_dir = directories
            args.complete_annotation = True
            result = audit(args)
            self.assertEqual(result['counts']['common_val_pixels'], 16)
            self.assertEqual(result['counts']['potential_validation'], 8)
            self.assertEqual(result['counts']['train_pixels'], 24)
            self.assertTrue(result['annotation_completeness_confirmed'])
            self.assertFalse(result['evaluation_ready'])
            self.assertEqual(len(result['splits']), 2)


if __name__ == '__main__':
    unittest.main()
