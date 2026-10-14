import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments import evaluate
from experiments.metrics import boundary_counts, instance_counts, pixel_counts, scores
from experiments.raster import check_alignment, index_files, read_labels, read_rgb, save_labels
from experiments.refine import generate_points, interior_point, refine_image


def options(**kwargs):
    defaults = dict(mode='interior', residual_policy='attach', box_pad=2, support_pad=0, min_area=3,
                    superpixel_area=25, compactness=10, min_radius=1, max_points=8,
                    min_score=0.8, min_stability=0.9, stability_offset=1,
                    min_support=0.8, min_coverage=0.5, max_overlap=0.05)
    return SimpleNamespace(**(defaults | kwargs))


class FakePredictor:
    model = SimpleNamespace(mask_threshold=0.0)

    def __init__(self, masks, score=0.95):
        self.masks = masks
        self.score = score
        self.calls = []

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        logits = np.where(np.stack(self.masks), 4.0, -4.0)
        return logits, np.full(len(self.masks), self.score), np.zeros((len(self.masks), 256, 256))


class MetricsTests(unittest.TestCase):
    def counts(self, pred, gt):
        return {**pixel_counts(pred, gt), **instance_counts(pred, gt), **boundary_counts(pred, gt, 0)}

    def test_perfect_with_unrelated_ids(self):
        gt = np.array([[0, 1, 1, 0], [0, 2, 2, 0]])
        result = scores(self.counts(gt * 17, gt))
        for value in result.values():
            self.assertEqual(value, 1.0)

    def test_merged_parcels_semantics_perfect_instances_worse(self):
        gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        result = scores(self.counts(np.ones_like(gt), gt))
        self.assertEqual(result['mask_iou'], 1)
        self.assertAlmostEqual(result['instance_f1'], 2 / 3)
        self.assertAlmostEqual(result['pq'], 1 / 3)
        self.assertEqual(result['boundary_f1'], 0)

    def test_empty_and_sparse_large_ids(self):
        empty = np.zeros((3, 3), dtype=np.uint32)
        self.assertEqual(scores(self.counts(empty, empty))['pq'], 0)
        labels = empty.copy()
        labels[1, 1] = 2 ** 31
        self.assertEqual(instance_counts(labels, labels)['instance_tp'], 1)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.original = np.zeros((20, 20), dtype=np.uint16)
        self.original[3:17, 3:17] = 5
        self.rgb = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_interior_point_never_selects_background_for_border_object(self):
        mask = np.ones((5, 5), dtype=bool)
        point, radius = interior_point(mask)
        np.testing.assert_equal(point, [2, 2])
        self.assertEqual(radius, 3)

    def test_duplicate_candidates_and_no_point_mask_fallback(self):
        region = self.original > 0
        predictor = FakePredictor([region, region, region])
        out, logs = refine_image(self.rgb, self.original, predictor, options())
        self.assertEqual(logs[0]['accepted_count'], 1)
        np.testing.assert_array_equal(out > 0, region)
        absent = np.zeros_like(region)
        absent[0:2, 0:2] = True
        out, logs = refine_image(self.rgb, self.original, FakePredictor([absent]), options())
        np.testing.assert_array_equal(out > 0, region)
        self.assertEqual(logs[0]['accepted_count'], 0)

    def test_two_candidates_split_merge_without_deleting_pixels(self):
        left = (self.original > 0).copy()
        left[:, 10:] = False
        right = (self.original > 0) & ~left
        predictor = FakePredictor([left, right])
        out, logs = refine_image(self.rgb, self.original, predictor, options(mode='box'))
        self.assertEqual(logs[0]['accepted_count'], 2)
        self.assertEqual(len(np.unique(out[out > 0])), 2)
        np.testing.assert_array_equal(out > 0, self.original > 0)
        self.assertIsNone(predictor.calls[0]['point_coords'])

    def test_slic_points_inside_source_and_capped(self):
        points = generate_points(self.rgb, self.original > 0, 'slic', options(max_points=3))
        self.assertTrue(1 <= len(points) <= 3)
        for x, y in points:
            self.assertEqual(self.original[int(y), int(x)], 5)

    def test_replace_can_remove_residual_without_creating_instances(self):
        candidate = np.zeros_like(self.original, dtype=bool)
        candidate[4:16, 4:16] = True
        out, logs = refine_image(self.rgb, self.original, FakePredictor([candidate]),
                                 options(residual_policy='replace'))
        np.testing.assert_array_equal(out > 0, candidate)
        self.assertEqual(logs[0]['removed_pixels'], 196 - 144)
        self.assertEqual(logs[0]['residual_components_attached'], 0)
        self.assertEqual(logs[0]['output_instance_count'], 1)

    def test_replace_low_coverage_and_rejected_candidates_keep_original(self):
        candidate = np.zeros_like(self.original, dtype=bool)
        candidate[8:12, 8:12] = True
        for score, threshold in [(0.95, 0.5), (0.1, 0.0)]:
            out, logs = refine_image(self.rgb, self.original, FakePredictor([candidate], score),
                                     options(residual_policy='replace', min_coverage=threshold))
            np.testing.assert_array_equal(out > 0, self.original > 0)
            self.assertEqual(logs[0]['removed_pixels'], 0)

    def test_replace_does_not_overwrite_neighbor_instance(self):
        self.original[3:17, 17:19] = 8
        candidate = self.original > 0
        out, logs = refine_image(self.rgb, self.original, FakePredictor([candidate]),
                                 options(residual_policy='replace', mode='box', support_pad=3))
        self.assertNotEqual(out[5, 5], out[5, 18])
        self.assertTrue(np.all(out[self.original == 8] > 0))

    def test_multiple_residual_fragments_do_not_become_new_instances(self):
        # A centered cross leaves four disconnected corner residuals. The previous
        # algorithm turned one source parcel into five predicted parcels.
        candidate = np.zeros_like(self.original, dtype=bool)
        candidate[3:17, 6:14] = True
        candidate[6:14, 3:17] = True
        out, logs = refine_image(self.rgb, self.original,
                                 FakePredictor([candidate]), options())
        np.testing.assert_array_equal(out > 0, self.original > 0)
        self.assertEqual(len(np.unique(out[out > 0])), 1)
        self.assertEqual(logs[0]['residual_components_attached'], 4)
        self.assertEqual(logs[0]['output_instance_count'], 1)

    def test_multi_candidate_residual_uses_only_existing_ids(self):
        left = np.zeros_like(self.original, dtype=bool)
        right = left.copy()
        left[4:16, 3:9] = True
        right[4:16, 11:17] = True
        out, logs = refine_image(self.rgb, self.original,
                                 FakePredictor([left, right]), options(mode='box'))
        self.assertEqual(logs[0]['accepted_count'], 2)
        self.assertEqual(len(np.unique(out[out > 0])), 2)
        np.testing.assert_array_equal(out > 0, self.original > 0)
        self.assertEqual(out[3, 4], out[5, 4])
        self.assertEqual(out[3, 15], out[5, 15])

    def test_other_objects_preserved_and_low_score_fallback(self):
        self.original[3:17, 17:19] = 8
        out, logs = refine_image(self.rgb, self.original,
                                 FakePredictor([self.original > 0], score=0.1), options(support_pad=3))
        np.testing.assert_array_equal(out > 0, self.original > 0)
        self.assertNotEqual(out[5, 5], out[5, 18])
        self.assertTrue(all(log['accepted_count'] == 0 for log in logs))


class RasterTests(unittest.TestCase):
    def test_georeference_nodata_and_cli_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gt, pred = root / 'gt', root / 'pred'
            profile = dict(driver='GTiff', height=8, width=8, count=1,
                           dtype='uint16', crs='EPSG:32648', transform=from_origin(100, 200, 1, 1))
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[2:6, 2:6] = 7
            save_labels(gt / '30.tif', labels, profile)
            save_labels(pred / '30_labels.tif', labels * 2, profile)
            with rasterio.open(gt / '30.tif', 'r+') as dst:
                dst.nodata = 0  # Matches existing BsiNet output convention.
            loaded, copied = read_labels(gt / '30.tif')
            np.testing.assert_array_equal(loaded, labels)
            self.assertEqual(copied['transform'], profile['transform'])
            self.assertEqual(index_files(pred, '_labels')['30'], pred / '30_labels.tif')
            (root / 'ids.txt').write_text('30\n', encoding='utf-8')
            argv = ['evaluate', '--gt-dir', str(gt), '--prediction-dir', str(pred),
                    '--prediction-suffix', '_labels', '--ids', str(root / 'ids.txt'),
                    '--output-dir', str(root / 'eval')]
            with patch('sys.argv', argv):
                evaluate.main()
            summary = json.loads((root / 'eval' / 'summary.json').read_text())
            self.assertEqual(summary['pq'], 1)
            self.assertTrue((root / 'eval' / 'metrics.csv').exists())
            wrong = copied.copy()
            wrong['transform'] = from_origin(101, 200, 1, 1)
            with self.assertRaises(ValueError):
                check_alignment(copied, wrong, 'first', 'second')
            with self.assertRaises(ValueError):
                read_rgb(gt / '30.tif', [1, 2, 3], None)

    def test_uint16_rgb_requires_explicit_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rgb.tif'
            profile = dict(driver='GTiff', height=4, width=4, count=3, dtype='uint16',
                           transform=from_origin(100, 200, 1, 1))
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(np.full((3, 4, 4), 1000, dtype=np.uint16))
            with self.assertRaises(ValueError):
                read_rgb(path, [1, 2, 3], None)
            rgb, _ = read_rgb(path, [1, 2, 3], [0, 2000])
            self.assertEqual(rgb.shape, (4, 4, 3))
            self.assertTrue(np.all(rgb == 127))


if __name__ == '__main__':
    unittest.main()
