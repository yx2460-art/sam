"""Independent evaluation; no ground truth enters SAM inference."""
import argparse
import csv
import json
from pathlib import Path

from .metrics import boundary_counts, instance_counts, pixel_counts, scores
from .raster import check_alignment, index_files, read_labels


def main():
    p = argparse.ArgumentParser(description='Compare instance-label maps on a fixed ID list')
    p.add_argument('--gt-dir', type=Path, required=True)
    p.add_argument('--prediction-dir', type=Path, required=True)
    p.add_argument('--prediction-suffix', default='')
    p.add_argument('--gt-suffix', default='')
    p.add_argument('--ids', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--boundary-tolerance', type=int, default=2)
    args = p.parse_args()
    if args.boundary_tolerance < 0:
        raise ValueError('Boundary tolerance must be nonnegative')
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    gt_files = index_files(args.gt_dir, args.gt_suffix)
    pred_files = index_files(args.prediction_dir, args.prediction_suffix)
    ids = [line.strip() for line in args.ids.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate evaluation IDs')
    rows, totals = [], {}
    for sample_id in ids:
        if sample_id not in gt_files or sample_id not in pred_files:
            raise ValueError(f'Missing evaluation pair: {sample_id}')
        target, gt_profile = read_labels(gt_files[sample_id])
        prediction, pred_profile = read_labels(pred_files[sample_id])
        check_alignment(gt_profile, pred_profile, gt_files[sample_id], pred_files[sample_id])
        counts = {**pixel_counts(prediction, target), **instance_counts(prediction, target),
                  **boundary_counts(prediction, target, args.boundary_tolerance)}
        rows.append({'id': sample_id, **scores(counts), **counts})
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    summary = {'sample_count': len(rows), 'ids': ids, 'aggregation': 'pooled_counts',
               'instance_match': 'greedy descending IoU >= 0.5, one-to-one',
               'boundary_tolerance_pixels': args.boundary_tolerance,
               'boundary_definition': '4-neighbor label transitions, foreground side; includes internal IDs',
               'empty_denominator_score': 0.0,
               'gt_dir': str(args.gt_dir.resolve()), 'prediction_dir': str(args.prediction_dir.resolve()),
               **scores(totals), 'counts': totals}
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
