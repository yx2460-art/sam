"""Summarize boundary/edge deletion patterns from existing offline diagnostic CSV."""
import argparse
import csv
from pathlib import Path


def percent(numerator, denominator):
    return f'{100*numerator/denominator:.2f}%' if denominator else 'N/A'


def summarize(diagnosis_dir):
    directory=Path(diagnosis_dir)
    with (directory/'changes.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    lines=['Offline location analysis relative to union; GT used only after inference.',
           'harmful=removed true foreground; correct=removed false foreground.',
           'share=share of all harmful deletions; rate=harmful deletions / all pixels in this location.',
           'Rate uses all location pixels, not only original true foreground.',
           'GT-boundary and window-edge groups overlap; do not add across classifications.']
    keys=('pixels','removed_true_foreground','removed_false_foreground',
          'added_true_foreground','added_false_foreground','relabeled_foreground')
    for method in ('conflict','random'):
        selected=[r for r in rows if r['method']==method and r['region']=='all_roi']
        locations={r['location']:r for r in selected}
        expected={'all','near_gt_boundary','away_from_gt_boundary',
                  'near_window_edge','away_from_window_edge'}
        if len(selected)!=5 or set(locations)!=expected:
            raise ValueError(f'{method}: expected five unique all_roi location rows')
        for row in selected:
            for key in keys:
                row[key]=int(row[key])
                if row[key]<0:
                    raise ValueError('Negative diagnostic count')
        total=locations['all']
        for axis in ('gt_boundary','window_edge'):
            near,away=locations[f'near_{axis}'],locations[f'away_from_{axis}']
            if any(near[k]+away[k]!=total[k] for k in keys):
                raise ValueError(f'{method}: {axis} groups do not partition all_roi')
        lines.append(f"{method}: harmful={total['removed_true_foreground']}, correct={total['removed_false_foreground']}, "
                     f"harmful among removals={percent(total['removed_true_foreground'],total['removed_true_foreground']+total['removed_false_foreground'])}")
        for name in ('near_gt_boundary','away_from_gt_boundary','near_window_edge','away_from_window_edge'):
            row=locations[name]
            lines.append(f"  {name}: pixels={row['pixels']}, harmful={row['removed_true_foreground']}, "
                         f"correct={row['removed_false_foreground']}, "
                         f"share={percent(row['removed_true_foreground'],total['removed_true_foreground'])}, "
                         f"rate={percent(row['removed_true_foreground'],row['pixels'])}")
    text='\n'.join(lines)+'\n'
    output=directory/'location_summary.txt'
    if output.exists():
        raise FileExistsError(f'Summary already exists: {output}')
    output.write_text(text,encoding='utf-8')
    print(text)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis-dir',type=Path,required=True)
    summarize(parser.parse_args().diagnosis_dir)


if __name__=='__main__':
    main()
