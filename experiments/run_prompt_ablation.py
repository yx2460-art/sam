"""Run using python -m experiments.run_prompt_ablation --help."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .raster import check_alignment, index_files, read_labels, read_rgb, save_labels


def parser():
    p = argparse.ArgumentParser(description='BsiNet-guided SAM prompt ablation (no GT inputs)')
    p.add_argument('--image-dir', type=Path, required=True)
    p.add_argument('--instance-dir', type=Path, required=True)
    p.add_argument('--instance-suffix', default='', help='Use _labels for BsiNet instance_pred output')
    p.add_argument('--ids', type=Path, required=True, help='Fixed one-ID-per-line validation/test list')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='New, nonexistent experiment directory')
    p.add_argument('--mode', choices=['box', 'interior', 'slic'], required=True)
    p.add_argument('--model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rgb-bands', type=int, nargs=3, default=[1, 2, 3])
    p.add_argument('--rgb-range', type=float, nargs=2)
    p.add_argument('--box-pad', type=int, default=2)
    p.add_argument('--support-pad', type=int, default=2)
    p.add_argument('--min-area', type=int, default=30)
    p.add_argument('--superpixel-area', type=int, default=400)
    p.add_argument('--compactness', type=float, default=10)
    p.add_argument('--min-radius', type=float, default=3, help='Pixels, not meters')
    p.add_argument('--max-points', type=int, default=16)
    p.add_argument('--min-score', type=float, default=0.8)
    p.add_argument('--min-stability', type=float, default=0.9)
    p.add_argument('--stability-offset', type=float, default=1)
    p.add_argument('--min-support', type=float, default=0.8)
    p.add_argument('--min-coverage', type=float, default=0.5)
    p.add_argument('--max-overlap', type=float, default=0.05)
    return p


def main():
    args = parser().parse_args()
    for name in ('min_score', 'min_stability', 'min_support', 'min_coverage', 'max_overlap'):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    if min(args.min_area, args.superpixel_area, args.max_points, args.compactness,
           args.stability_offset, args.min_radius) <= 0 or min(args.box_pad, args.support_pad) < 0:
        raise ValueError('Invalid area, point, distance or padding parameters')
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    ids = [line.strip() for line in args.ids.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('ID list is empty or contains duplicates')
    images = index_files(args.image_dir)
    instances = index_files(args.instance_dir, args.instance_suffix)
    for sample_id in ids:
        if sample_id not in images or sample_id not in instances:
            raise ValueError(f'Missing image or instance map for {sample_id}')
        # Validate every selected pair before loading the model or creating outputs.
        rgb, profile = read_rgb(images[sample_id], args.rgb_bands, args.rgb_range)
        _, instance_profile = read_labels(instances[sample_id])
        check_alignment(profile, instance_profile, images[sample_id], instances[sample_id])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'segment-anything'))
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from .refine import refine_image
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly')
    model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    model.to(device=args.device)
    predictor = SamPredictor(model)
    import numpy
    import rasterio
    import scipy
    import skimage
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], check=False)
    args.output_dir.mkdir(parents=True)
    manifest = {'ground_truth_used': False, 'ids': ids,
                'checkpoint_sha256': checkpoint_hash,
                'code_revision': revision.stdout.strip() if revision.returncode == 0 else None,
                'parameters': {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': numpy.__version__, 'rasterio': rasterio.__version__,
                             'scipy': scipy.__version__, 'scikit-image': skimage.__version__},
                'samples': []}
    for sample_id in ids:
        rgb, profile = read_rgb(images[sample_id], args.rgb_bands, args.rgb_range)
        original, _ = read_labels(instances[sample_id])
        start = time.perf_counter()
        with torch.inference_mode():
            refined, records = refine_image(rgb, original, predictor, args)
        seconds = time.perf_counter() - start
        save_labels(args.output_dir / 'instance' / f'{sample_id}.tif', refined, profile)
        sample = {'id': sample_id, 'seconds': seconds, 'objects': records}
        (args.output_dir / f'{sample_id}_prompts.json').write_text(
            json.dumps(sample, indent=2), encoding='utf-8')
        manifest['samples'].append({'id': sample_id, 'seconds': seconds})
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{sample_id}: {seconds:.2f}s; saved {args.output_dir / "instance"}')


if __name__ == '__main__':
    main()
