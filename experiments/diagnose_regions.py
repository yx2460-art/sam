"""Offline pixel-transition audit of completed region experiments; never loads SAM."""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .metrics import label_boundary
from .run_tiles import aligned, digest


def transitions(before, after, truth, mask):
    p, q, t = before > 0, after > 0, truth > 0
    removed, added = p & ~q & mask, ~p & q & mask
    return dict(pixels=int(mask.sum()),
                removed_true_foreground=int((removed & t).sum()),
                removed_false_foreground=int((removed & ~t).sum()),
                added_true_foreground=int((added & t).sum()),
                added_false_foreground=int((added & ~t).sum()),
                relabeled_foreground=int((p & q & (before != after) & mask).sum()))


def diagnose(run_dir, gt_path, output_dir, radius=2):
    run_dir, gt_path, output_dir = map(Path, (run_dir, gt_path, output_dir))
    if radius < 0:
        raise ValueError('Radius must be nonnegative')
    if output_dir.exists():
        raise FileExistsError(f'Use a new directory: {output_dir}')
    report_path = run_dir/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Expected completed region experiment')
    union_dir = Path(report['source_union'])
    destination = output_dir.resolve()
    for protected in (run_dir.resolve(), union_dir.resolve()):
        if destination == protected or protected in destination.parents:
            raise ValueError('Diagnosis output must be outside the source experiment directories')
    manifest_path = union_dir/'manifest.json'
    if digest(manifest_path) != report['source_manifest_sha256']:
        raise ValueError('Union manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    params = manifest['parameters']
    source_path = Path(manifest['source_run'])/'manifest.json'
    if digest(source_path) != manifest['source_manifest_sha256']:
        raise ValueError('Window manifest changed')
    if digest(union_dir/'instance.tif') != report['source_union_sha256']:
        raise ValueError('Union raster changed')
    if report.get('gt_sha256') and digest(gt_path) != report['gt_sha256']:
        raise ValueError('GT differs from original evaluation')
    for key in ('prediction', 'eval_mask'):
        if digest(params[key]) != manifest['hashes'][key]:
            raise ValueError(f'Changed input: {key}')
    x0, y0, x1, y1 = manifest['crop_xyxy']
    window = Window(x0, y0, x1-x0, y1-y0)
    allow = params.get('allow_crs_mismatch', False)
    inputs = [report_path, manifest_path, source_path, gt_path,
              Path(params['eval_mask']), union_dir/'instance.tif',
              Path(manifest['source_run'])/'windows.jsonl']
    rows = []
    with ExitStack() as stack:
        union_src = stack.enter_context(rasterio.open(union_dir/'instance.tif'))
        full = stack.enter_context(rasterio.open(params['prediction']))
        gt = stack.enter_context(rasterio.open(gt_path))
        roi = stack.enter_context(rasterio.open(params['eval_mask']))
        aligned(full, gt, allow)
        aligned(full, roi, allow)
        if (union_src.shape != (y1-y0, x1-x0) or
                not union_src.transform.almost_equals(full.window_transform(window))):
            raise ValueError('Union crop grid mismatch')
        if not np.issubdtype(np.dtype(gt.dtypes[0]), np.integer) or gt.nodata not in (None, 0):
            raise ValueError('GT requires integer labels and NoData=None/0')
        target = gt.read(1, window=window)
        if np.any(target < 0):
            raise ValueError('Negative GT labels')
        initial = union_src.read(1)
        raw = roi.read(1, window=window)
        domain = (raw > 0) & np.isfinite(raw) & (roi.read_masks(1, window=window) > 0)

        def read_crop(path, binary=False):
            inputs.append(path)
            src = stack.enter_context(rasterio.open(path))
            aligned(union_src, src, allow)
            data = src.read(1)
            if not np.issubdtype(data.dtype, np.integer) or np.any(data < 0):
                raise ValueError(f'Invalid labels: {path}')
            if binary and np.any((data != 0) & (data != 1)):
                raise ValueError(f'Nonbinary mask: {path}')
            return data > 0 if binary else data

        conflict = read_crop(run_dir/'conflict.tif', True)
        if np.any(conflict & ~domain) or int(conflict.sum()) != report['total_conflict_pixels']:
            raise ValueError('Conflict mask disagrees with report/ROI')
        shape = initial.shape
        caches = {}
        for key in ('count', 'votes'):
            path = Path(manifest['source_run'])/'work'/f'{key}.bin'
            if path.stat().st_size != initial.size*4 or digest(path) != manifest['cache_sha256'][key]:
                raise ValueError(f'Vote cache changed: {key}')
            inputs.append(path)
            caches[key] = np.memmap(path,mode='r',dtype='uint32',shape=shape)
        expected_conflict = (caches['votes'] > 0) & (caches['votes'] < caches['count']) & domain
        if not np.array_equal(conflict,expected_conflict):
            raise ValueError('Conflict raster differs from original votes')
        # Include neighboring GT pixels outside the crop before detecting boundaries.
        halo = radius+1
        expanded = Window(x0-halo, y0-halo, x1-x0+2*halo, y1-y0+2*halo)
        boundary = label_boundary(gt.read(1, window=expanded, boundless=True, fill_value=0))
        near_boundary = ndimage.maximum_filter(boundary, size=2*radius+1, mode='constant')[halo:-halo, halo:-halo] > 0
        edges = np.zeros(domain.shape, dtype=bool)
        window_log = Path(manifest['source_run'])/'windows.jsonl'
        for line in window_log.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            left, top = item['x']-x0, item['y']-y0
            right, bottom = left+item['width'], top+item['height']
            for y in (top, bottom-1):
                if 0 <= y < edges.shape[0]:
                    edges[y, max(0,left):min(edges.shape[1],right)] = True
            for x in (left, right-1):
                if 0 <= x < edges.shape[1]:
                    edges[max(0,top):min(edges.shape[0],bottom), x] = True
        near_edge = ndimage.maximum_filter(edges, size=2*radius+1, mode='constant') > 0
        for method in ('conflict', 'random'):
            write = read_crop(run_dir/f'{method}_write.tif', True)
            after = read_crop(run_dir/method/'instance.tif')
            if np.any(write & ~domain) or np.any((initial != after) & ~write):
                raise ValueError(f'{method}: unexpected writes outside planned ROI')
            if int(write.sum()) != report['write_pixels_per_method']:
                raise ValueError(f'{method}: write mask budget changed')
            actual = dict(changed_pixels=int((initial != after).sum()),
                          added_pixels=int(((initial == 0) & (after > 0)).sum()),
                          removed_pixels=int(((initial > 0) & (after == 0)).sum()))
            if any(actual[k] != report['results'][method][k] for k in actual):
                raise ValueError(f'{method}: raster differs from recorded counts')
            groups = dict(all_roi=domain, raw_conflict=conflict,
                          selected_raw_conflict=write & conflict,
                          write_outside_raw_conflict=write & ~conflict,
                          outside_write=domain & ~write)
            for name, mask in groups.items():
                for location, subset in [('all',mask), ('near_gt_boundary',mask & near_boundary),
                                         ('away_from_gt_boundary',mask & ~near_boundary),
                                         ('near_window_edge',mask & near_edge),
                                         ('away_from_window_edge',mask & ~near_edge)]:
                    rows.append(dict(method=method, region=name, location=location,
                                     **transitions(initial,after,target,subset)))
    snapshot = {str(path.resolve()): digest(path) for path in inputs}
    result = dict(offline_only=True, ground_truth_used_for='post-hoc diagnosis only',
                  training_independence_confirmed=False, radius_pixels=radius,
                  proximity='Chebyshev distance; GT instance boundaries; actual first-stage window perimeters',
                  warning='Overlapping location groups must not be added together. Pixel foreground correctness does not measure instance identity.',
                  input_sha256=snapshot, rows=rows)
    output_dir.mkdir(parents=True)
    (output_dir/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output_dir/'changes.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compact = [r for r in rows if r['location']=='all' and r['region'] in
               ('all_roi','selected_raw_conflict','write_outside_raw_conflict')]
    lines = ['Offline diagnostic: counts relative to union; no SAM inference.']
    for row in compact:
        lines.append(f"{row['method']} / {row['region']}: harmful removals={row['removed_true_foreground']}, "
                     f"correct removals={row['removed_false_foreground']}, "
                     f"correct additions={row['added_true_foreground']}, false additions={row['added_false_foreground']}")
    text='\n'.join(lines)+'\n'
    (output_dir/'summary.txt').write_text(text,encoding='utf-8')
    print(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--radius',type=int,default=2)
    args=parser.parse_args()
    diagnose(args.run_dir,args.gt,args.output_dir,args.radius)


if __name__=='__main__':
    main()
