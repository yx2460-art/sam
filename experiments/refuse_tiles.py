"""Re-fuse a completed run's cached votes; no model loading or SAM inference."""
import argparse
import contextlib
import copy
import csv
import io
import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import Window

from .run_tiles import aligned, digest
from .tiles import fuse_result


METHODS = ('center', 'union', 'intersection', 'majority')


def blocks(shape):
    for y in range(0, shape[0], 512):
        for x in range(0, shape[1], 512):
            h, w = min(512, shape[0]-y), min(512, shape[1]-x)
            yield Window(x, y, w, h), np.s_[y:y+h, x:x+w]


def refusion(run_dir, output_dir, gt_path=None, methods=METHODS):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Choose unique fusion methods: ' + ', '.join(METHODS))
    if output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {output_dir}')
    if run_dir == output_dir or run_dir in output_dir.parents:
        raise ValueError('Output must be outside the source run')
    manifest_path = run_dir / 'manifest.json'
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    if source.get('status') != 'complete':
        raise ValueError('Only a complete run can be re-fused; smoke/failed runs are rejected')
    if source.get('instance_identity') != 'fixed BsiNet source ID; no new parcel split':
        raise ValueError('Unsupported instance protocol')
    params = source['parameters']
    baseline_path = Path(params['prediction']).resolve()
    mask_path = Path(params['eval_mask']).resolve()
    image_path = Path(params['image']).resolve()
    coverage_path = Path(params['coverage']).resolve()
    if digest(baseline_path) != source['hashes']['prediction']:
        raise ValueError('BsiNet prediction differs from the original run')
    if digest(mask_path) != source['hashes']['eval_mask']:
        raise ValueError('Scoring mask differs from the original run')
    x0, y0, x1, y1 = source['crop_xyxy']
    shape = (y1-y0, x1-x0)
    if min(shape) <= 0:
        raise ValueError('Invalid crop bounds')
    cache_paths = {key:run_dir/'work'/f'{key}.bin' for key in ('count','votes','center')}
    caches = {}
    for key, path in cache_paths.items():
        dtype = np.dtype('uint8' if key == 'center' else 'uint32')
        if not path.is_file() or path.stat().st_size != int(np.prod(shape))*dtype.itemsize:
            raise ValueError(f'Missing or wrong-sized cache: {path}; no inference was performed')
        caches[key] = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    count, votes, center = (caches[k] for k in ('count','votes','center'))
    cache_hashes = {key:digest(path) for key,path in cache_paths.items()}
    records_path = run_dir/'windows.jsonl'
    record_count = 0
    recorded_valid = 0
    seen = set()
    with records_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['x'],row['y'],row['width'],row['height'])
            if row['index'] != record_count or key in seen:
                raise ValueError('Duplicate or nonsequential window records')
            seen.add(key)
            if not (x0 <= row['x'] < row['x']+row['width'] <= x1 and
                    y0 <= row['y'] < row['y']+row['height'] <= y1):
                raise ValueError('Window record outside the stored crop')
            tile = (run_dir/row['tile']).resolve()
            if run_dir not in tile.parents or not tile.is_file():
                raise ValueError(f'Missing/out-of-run saved tile: {tile}')
            recorded_valid += int(row['valid_pixels'])
            record_count += 1
    if record_count != source['processed_windows'] or record_count != source['requested_windows']:
        raise ValueError('Window records are incomplete')
    allow = bool(params.get('allow_crs_mismatch',False))
    with rasterio.open(image_path) as image, rasterio.open(baseline_path) as base, \
         rasterio.open(mask_path) as roi, rasterio.open(coverage_path) as coverage, \
         rasterio.open(run_dir/'count.tif') as stored_count, \
         rasterio.open(run_dir/'instance.tif') as stored_instance:
        for src in (base,roi,coverage):
            aligned(image,src,allow)
        expected_transform = image.window_transform(Window(x0,y0,shape[1],shape[0]))
        expected_grid = SimpleNamespace(width=shape[1],height=shape[0],
                                        crs=image.crs,transform=expected_transform)
        cached_crs_overrides = []
        for src in (stored_count,stored_instance):
            try:
                issues = aligned(expected_grid,src,allow)
            except ValueError as error:
                raise ValueError(f'Cached output grid mismatch: {error}') from error
            if issues:
                cached_crs_overrides.append(src.name)
        stats = dict(scoring_pixels=0,scoring_overlap_pixels=0,scoring_conflict_pixels=0,
                     observed_pixels=0,total_observations=0)
        for win, sl in blocks(shape):
            c,v,z = count[sl],votes[sl],center[sl]
            if np.any(v>c) or np.any(z>1) or np.any((c==0)&(z!=0)):
                raise ValueError('Invalid count/votes/center cache values')
            if np.any((z>0)&(v==0)) or np.any((c>0)&(v==c)&(z==0)):
                raise ValueError('Center decision contradicts cached observations')
            if not np.array_equal(c,stored_count.read(1,window=win)):
                raise ValueError('Cached counts differ from count.tif')
            full_win = Window(x0+win.col_off,y0+win.row_off,win.width,win.height)
            original = base.read(1,window=full_win)
            if not np.array_equal(fuse_result(original,v,c,z,source['fusion']),
                                  stored_instance.read(1,window=win)):
                raise ValueError('Caches do not reproduce the original fused instance raster')
            raw = roi.read(1,window=full_win)
            selected = (raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=full_win)>0)
            if np.any(selected & (c==0)):
                raise ValueError('Fixed scoring mask has uncovered pixels')
            stats['scoring_pixels'] += int(selected.sum())
            stats['scoring_overlap_pixels'] += int((selected&(c>1)).sum())
            stats['scoring_conflict_pixels'] += int((selected&(v>0)&(v<c)).sum())
            stats['observed_pixels'] += int((c>0).sum())
            stats['total_observations'] += int(c.sum(dtype=np.uint64))
        if stats['total_observations'] != recorded_valid:
            raise ValueError('Observation count differs from window log')
        if not stats['scoring_pixels']:
            raise ValueError('Empty scoring area')
        output_dir.mkdir(parents=True)
        revision = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,
                                  cwd=Path(__file__).resolve().parents[1],check=False).stdout.strip()
        report = dict(status='running',source_run=str(run_dir),sam_inference_performed=False,
                      cached_output_crs_overrides=cached_crs_overrides,
                      crs_equivalence_confirmed=False,
                      source_manifest_sha256=digest(manifest_path),
                      source_windows_sha256=digest(records_path),cache_sha256=cache_hashes,
                      cache_integrity_scope='sizes, invariants, original output reproduction and log totals; '
                                            'original run did not record cache hashes at creation',
                      source_inference_seconds=source.get('seconds'),source_code_revision=source.get('code_revision'),
                      refusion_code_revision=revision,methods=list(methods),stats=stats,
                      notes='Fixed source instance IDs; binary mask fusion, not SAM probability fusion. '
                            'ROI-clipped development metrics, training independence unverified.',results={})
        rows = []
        if gt_path is not None:
            gt_path = Path(gt_path).resolve()
            report['gt_sha256'] = digest(gt_path)
        try:
            for method in methods:
                tick = time.perf_counter()
                directory = output_dir/method
                directory.mkdir()
                profile = stored_instance.profile.copy()
                with rasterio.open(directory/'instance.tif','w',**profile) as dst:
                    for win,sl in blocks(shape):
                        original = base.read(1,window=Window(x0+win.col_off,y0+win.row_off,win.width,win.height))
                        dst.write(fuse_result(original,votes[sl],count[sl],center[sl],method),1,window=win)
                fusion_seconds = time.perf_counter()-tick
                shutil.copyfile(run_dir/'count.tif',directory/'count.tif')
                derived = copy.deepcopy(source)
                derived['parameters'].update(image=str(image_path),prediction=str(baseline_path),
                                             coverage=str(coverage_path),eval_mask=str(mask_path),
                                             output_dir=str(directory),fusion=method,
                                             gt=None if gt_path is None else str(gt_path))
                derived.update(fusion=method,operation='cached_refusion',source_run=str(run_dir),
                               cached_output_crs_overrides=cached_crs_overrides,
                               sam_inference_performed=False,source_inference_seconds=source.get('seconds'),
                               seconds=fusion_seconds,code_revision=revision,
                               source_manifest_sha256=report['source_manifest_sha256'],
                               cache_sha256=cache_hashes)
                (directory/'manifest.json').write_text(json.dumps(derived,indent=2),encoding='utf-8')
                entry = dict(fusion_seconds=fusion_seconds)
                if gt_path is not None:
                    from .evaluate_tiles import evaluate_run
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary = evaluate_run(directory,gt_path,mask_path,allow,
                                               params.get('boundary_tolerance',2))
                    if not rows:
                        baseline = {k:v for k,v in summary['baseline'].items() if k!='counts'}
                        rows.append(dict(method='baseline',fusion_seconds=0.0,**baseline))
                    metrics = {k:v for k,v in summary['sam'].items() if k!='counts'}
                    rows.append(dict(method=method,fusion_seconds=fusion_seconds,**metrics))
                    entry.update(summary=summary)
                report['results'][method] = entry
                print(f'{method}: saved {directory}; fusion {fusion_seconds:.2f}s',flush=True)
            # A concurrent change must not silently invalidate the comparison.
            if any(digest(path)!=cache_hashes[key] for key,path in cache_paths.items()):
                raise ValueError('Source caches changed during re-fusion')
            if digest(manifest_path)!=report['source_manifest_sha256'] or digest(records_path)!=report['source_windows_sha256']:
                raise ValueError('Source run records changed during re-fusion')
            report['status'] = 'complete'
            if rows:
                with (output_dir/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
        except Exception as error:
            report.update(status='failed',error=str(error))
            raise
        finally:
            (output_dir/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gt',type=Path,help='Optional GT for automatic evaluation after fusion')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    args = p.parse_args()
    refusion(args.run_dir,args.output_dir,args.gt,args.methods)


if __name__=='__main__':
    main()
