"""Read-only full-raster audit; never infer annotation completeness from ID zero."""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


def grid_errors(reference, source):
    errors = []
    if (source.width, source.height) != (reference.width, reference.height):
        errors.append('raster dimensions differ')
    if reference.crs is None or source.crs is None:
        errors.append('missing CRS; alignment cannot be certified')
    elif source.crs != reference.crs:
        errors.append('CRS differs')
    # Relative affine.almost_equals tolerances can be misleading at large origins.
    if not np.allclose(tuple(reference.transform), tuple(source.transform), rtol=0, atol=1e-8):
        errors.append('affine pixel grid differs')
    return errors


def load_split(directory, tile_size, width, height):
    directory = Path(directory)
    def ids(name):
        values = [Path(x.strip()).stem for x in (directory / name).read_text(
            encoding='utf-8-sig').splitlines() if x.strip()]
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Empty/duplicate IDs: {name}')
        return set(values)
    train, val = ids('train_ids.txt'), ids('val_ids.txt')
    if train & val:
        raise ValueError('Train and validation ID lists overlap')
    rows = {}
    with (directory / 'tiles_index.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = Path(row['filename']).stem
            if key in rows:
                raise ValueError(f'Duplicate index ID: {key}')
            x, y = int(row['x']), int(row['y'])
            if x < 0 or y < 0 or x + tile_size > width or y + tile_size > height:
                raise ValueError(f'Index window out of bounds: {key}')
            rows[key] = (x, y, x + tile_size, y + tile_size)
    missing = (train | val) - rows.keys()
    if missing:
        raise ValueError(f'IDs missing from index: {sorted(missing)[:10]}')
    return ([rows[key] for key in sorted(train)], [rows[key] for key in sorted(val)])


def footprint(rectangles, window, buffer=0):
    x, y, w, h = map(int, (window.col_off, window.row_off, window.width, window.height))
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        left, top = max(x, x0 - buffer), max(y, y0 - buffer)
        right, bottom = min(x + w, x1 + buffer), min(y + h, y1 + buffer)
        if right > left and bottom > top:
            mask[top-y:bottom-y, left-x:right-x] = True
    return mask


def metadata(src):
    return dict(path=src.name, width=src.width, height=src.height, bands=src.count,
                dtypes=list(src.dtypes), crs=str(src.crs), transform=list(src.transform),
                nodata=None if src.nodata is None else str(src.nodata))


def audit(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    report = {'status': 'needs_review', 'errors': [], 'warnings': [], 'rasters': {},
              'annotation_completeness_confirmed': False,
              'evaluation_ready': False, 'counts': {}}
    with ExitStack() as stack:
        paths = {'image': args.image, 'gt': args.gt, 'prediction': args.prediction,
                 'coverage': args.coverage}
        if args.annotation_roi:
            paths['annotation_roi'] = args.annotation_roi
        sources = {key: stack.enter_context(rasterio.open(path)) for key, path in paths.items()}
        ref = sources['image']
        for key, src in sources.items():
            report['rasters'][key] = metadata(src)
            report['errors'].extend(f'{key}: {error}' for error in grid_errors(ref, src))
            if key != 'image' and src.count != 1:
                report['errors'].append(f'{key}: expected a one-band raster')
            if key in {'gt', 'prediction'}:
                if not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                    report['errors'].append(f'{key}: instance IDs must be integer')
                if src.nodata not in (None, 0):
                    report['errors'].append(f'{key}: unsupported label NoData, expected None or 0')
        if ref.count < 3:
            report['errors'].append('image: expected at least three RGB bands; inspect band order')
        train, val = [], []
        if args.split_dir:
            try:
                train, val = load_split(args.split_dir, args.tile_size, ref.width, ref.height)
                report['split'] = {'directory': str(args.split_dir), 'train_tiles': len(train),
                                   'val_tiles': len(val), 'tile_size': args.tile_size,
                                   'train_buffer_pixels': args.train_buffer,
                                   'footprint_mode': 'listed rectangles only',
                                   'matches_checkpoint_training_split_confirmed': False}
            except (ValueError, KeyError, OSError) as error:
                report['errors'].append(f'split: {error}')
        else:
            report['warnings'].append('No training/validation split supplied; held-out area is unverified.')
        summary = args.prediction.parent / 'full_inference_summary.json'
        if summary.exists():
            info = json.loads(summary.read_text(encoding='utf-8-sig'))
            report['prediction_provenance'] = info
            if info.get('image') and Path(info['image']).resolve() != args.image.resolve():
                report['warnings'].append('Summary source image path differs; verify input provenance.')
            output = info.get('outputs', {}).get('instance')
            if output and Path(output).resolve() != args.prediction.resolve():
                report['warnings'].append('Summary instance path differs; verify provenance.')
        else:
            report['warnings'].append('Prediction summary absent; model and inference provenance unverified.')
        report['warnings'].append('GT ID 0 is background, not proof of complete annotation.')
        if not args.annotation_roi:
            report['warnings'].append('No confirmed annotation ROI supplied; no evaluation area is approved.')
        args.output_dir.mkdir(parents=True)
        if not report['errors']:
            profile = dict(driver='GTiff', width=ref.width, height=ref.height, count=1,
                           dtype='uint8', crs=ref.crs, transform=ref.transform,
                           compress='deflate', nodata=None)
            potential = stack.enter_context(rasterio.open(
                args.output_dir / 'potential_validation.tif', 'w', **profile)) if val else None
            names = ('pixels', 'image_valid', 'coverage', 'coverage_on_valid', 'gt_foreground',
                     'gt_outside_coverage', 'prediction_foreground', 'prediction_outside_coverage',
                     'train_pixels', 'val_pixels', 'train_val_overlap', 'potential_validation')
            counts = dict.fromkeys(names, 0)
            for y in range(0, ref.height, 512):
                for x in range(0, ref.width, 512):
                    window = Window(x, y, min(512, ref.width-x), min(512, ref.height-y))
                    valid = np.all(ref.read_masks([1, 2, 3], window=window) > 0, axis=0)
                    rgb = ref.read([1, 2, 3], window=window)
                    valid &= np.all(np.isfinite(rgb), axis=0)
                    gt = sources['gt'].read(1, window=window)
                    pred = sources['prediction'].read(1, window=window)
                    cov = sources['coverage'].read(1, window=window)
                    if np.any(gt < 0) or np.any(pred < 0):
                        report['errors'].append('Negative instance IDs found')
                    if not np.all(np.isfinite(cov)) or not np.all(np.isin(cov, [0, 1, 255])):
                        report['errors'].append('Coverage must contain only 0/1 or 0/255')
                    covered = cov > 0
                    tr, va = footprint(train, window), footprint(val, window)
                    candidate = valid & covered & va & ~footprint(train, window, args.train_buffer)
                    if args.annotation_roi:
                        roi = sources['annotation_roi'].read(1, window=window)
                        candidate &= (roi > 0) & np.isfinite(roi) & (
                            sources['annotation_roi'].read_masks(1, window=window) > 0)
                    values = [valid.size, valid.sum(), covered.sum(), (covered & valid).sum(),
                              (gt > 0).sum(), ((gt > 0) & ~covered).sum(), (pred > 0).sum(),
                              ((pred > 0) & ~covered).sum(), tr.sum(), va.sum(), (tr & va).sum(),
                              candidate.sum()]
                    for key, value in zip(names, values):
                        counts[key] += int(value)
                    if potential:
                        potential.write(candidate.astype(np.uint8), 1, window=window)
            report['counts'] = counts
            report['coverage_fraction_of_grid'] = counts['coverage'] / counts['pixels']
            if counts['train_val_overlap']:
                report['warnings'].append('Training and validation footprints share pixels; excluded from potential area.')
            if val and not counts['potential_validation']:
                report['warnings'].append('No potential validation pixels remain after exclusions.')
            # Downsample a georeferenced diagnostic, not an evaluation mask.
            scale = min(1, 1400 / max(ref.width, ref.height))
            h, w = max(1, round(ref.height*scale)), max(1, round(ref.width*scale))
            fg = sources['gt'].read(1, out_shape=(h, w), resampling=Resampling.nearest) > 0
            cov = sources['coverage'].read(1, out_shape=(h, w), resampling=Resampling.nearest) > 0
            preview = fg.astype(np.uint8) + 2 * cov.astype(np.uint8)
            profile.update(width=w, height=h, transform=ref.transform * rasterio.Affine.scale(
                ref.width/w, ref.height/h))
            with rasterio.open(args.output_dir / 'preview.tif', 'w', **profile) as dst:
                dst.write(preview, 1)
                dst.write_colormap(1, {0:(40,40,40,255), 1:(255,0,0,255),
                                      2:(80,140,220,255), 3:(0,220,80,255)})
            report['preview_legend'] = {'0':'no coverage / no GT foreground',
                                        '1':'GT foreground outside coverage',
                                        '2':'covered background', '3':'covered GT foreground'}
            report['status'] = 'grid_checks_passed_review_required' if not report['errors'] else 'failed'
        else:
            report['status'] = 'failed'
    report['errors'] = sorted(set(report['errors']))
    report['next_step'] = ('Confirm annotation completeness, actual checkpoint training split and fixed '
                           'evaluation region before running tile experiments. potential_validation.tif '
                           'is a diagnostic candidate, not an approved test mask.')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [f"Status: {report['status']}", 'Evaluation ready: False',
             *[f'ERROR: {x}' for x in report['errors']],
             *[f'WARNING: {x}' for x in report['warnings']],
             json.dumps(report['counts'], indent=2), report['next_step']]
    (args.output_dir / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'gt', 'prediction', 'coverage', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-dir', type=Path, help='Actual checkpoint training split directory')
    p.add_argument('--annotation-roi', type=Path, help='Confirmed completely labeled area, nonzero=inside')
    p.add_argument('--tile-size', type=int, default=256, help='Tile size used to create split index')
    p.add_argument('--train-buffer', type=int, default=256)
    args = p.parse_args()
    if args.tile_size <= 0 or args.train_buffer < 0:
        p.error('tile-size must be positive and train-buffer nonnegative')
    report = audit(args)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
