"""Plan a new region without GT; transfer approved edits without changing old experiments."""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from .manual_fusion import read
from .raster import save_labels
from .region_workflow import load,prepare as region_prepare,write_json
from .run_tiles import aligned,digest
from .split_instances import sealed


def plan(result,instance_id,output):
    result=Path(result).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new plan directory')
    if type(instance_id) is not int or instance_id<=0:raise ValueError('Positive source ID required')
    previous=sealed(result/'region_result.json');workflow=sealed(Path(previous['interaction_project'])/'workflow.json')
    experiment,m=load(workflow['experiment']);run=Path(m['source_run']);manifest=read(run/'manifest.json');params=manifest['parameters']
    if previous['experiment_sha256']!=digest(experiment/'experiment.json'):raise ValueError('Source result belongs to another experiment')
    box=None;pixels=0;missing=0
    with rasterio.open(params['prediction']) as src,rasterio.open(params['eval_mask']) as roi:
        aligned(src,roi,m['allow_crs_mismatch'])
        if digest(params['prediction'])!=manifest['hashes']['prediction'] or digest(params['eval_mask'])!=manifest['hashes']['eval_mask']:raise ValueError('Source inputs changed')
        for _,win in src.block_windows(1):
            target=src.read(1,window=win)==instance_id;yy,xx=np.where(target)
            if not xx.size:continue
            raw=roi.read(1,window=win);valid=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
            pixels+=int(target.sum());missing+=int((target&~valid).sum())
            b=[int(win.col_off+xx.min()),int(win.row_off+yy.min()),int(win.col_off+xx.max()+1),int(win.row_off+yy.max()+1)]
            box=b if box is None else [min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3])]
    if box is None:raise ValueError('Source ID absent from full source prediction')
    cx0,cy0,cx1,cy1=manifest['crop_xyxy'];cached=cx0<=box[0]<box[2]<=cx1 and cy0<=box[1]<box[3]<=cy1
    status='ready' if cached and missing==0 else 'cache_missing' if not cached else 'annotation_partial'
    report=dict(status=status,source_result=str(result),old_experiment=str(experiment),source_run=str(run),instance_id=instance_id,
        suggested_bounds_xywh=[box[0],box[1],box[2]-box[0],box[3]-box[1]],target_xyxy=box,
        old_region_xyxy=m['region_xyxy'],cached_crop_xyxy=manifest['crop_xyxy'],target_source_pixels=pixels,
        target_pixels_outside_annotation_roi=missing,ground_truth_read=False,
        warning='Source-instance bbox is a location proposal, not a verified complete real-parcel boundary. New region scores must stay separate.',
        input_sha256={str(p):digest(p) for p in [result/'region_result.json',experiment/'experiment.json',run/'manifest.json']})
    output.mkdir(parents=True);write_json(output/'plan.json',report)
    lines=[f'Status: {status}',f'Old region: {m["region_xyxy"]}',f'Source ID {instance_id} full bbox: {box}',
        f'Cached crop: {manifest["crop_xyxy"]}',f'Target pixels outside annotation ROI: {missing}/{pixels}',
        'No GT read. Do not expand or overwrite the old evaluation region.']
    if not cached:lines.append('New SAM window coverage is required before preparing this region; old cache cannot cover the full target bbox.')
    if missing:lines.append('Annotation ROI does not cover all source-target pixels. Obtain/verify valid annotations for the new area before full-target evaluation; do not treat unannotated pixels as background.')
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
    return report


def prepare(plan_dir,output):
    p=sealed(Path(plan_dir).resolve()/'plan.json');output=Path(output).resolve()
    if p['status']!='ready':raise ValueError('Plan is not ready; resolve cache/annotation coverage first')
    return region_prepare(p['source_run'],output,instance_id=p['instance_id'])


def initialize(plan_dir,experiment,score_dir,output):
    p=sealed(Path(plan_dir).resolve()/'plan.json');experiment,m=load(experiment);score_dir=Path(score_dir).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new transferred-result directory')
    if p['status']!='ready' or m['source_run']!=p['source_run'] or m['requested_instance_id']!=p['instance_id'] or m['region_xyxy']!=p['target_xyxy']:raise ValueError('New experiment does not match region plan')
    scores=sealed(score_dir/'scores.json')
    if scores['experiment_sha256']!=digest(experiment/'experiment.json') or 'interactive' in scores['evaluation']:raise ValueError('Use first fusion evaluation for new region')
    previous=Path(p['source_result']);prior=sealed(previous/'region_result.json');old_workflow=sealed(Path(prior['interaction_project'])/'workflow.json')
    _,old=load(old_workflow['experiment']);method=prior['start_method']
    if method not in m['candidates']:raise ValueError('Unknown original fusion method')
    with rasterio.open(m['candidates'][method]) as src,rasterio.open(previous/'region_result.tif') as old_src,rasterio.open(old['mask']) as old_roi:
        data=src.read(1);profile=src.profile;nx0,ny0,nx1,ny1=m['context_xyxy'];ox0,oy0,ox1,oy1=old['context_xyxy']
        expected=src.transform*Affine.translation(ox0-nx0,oy0-ny0)
        if not expected.almost_equals(old_src.transform) or old_src.crs!=src.crs:raise ValueError('Transfer grid mismatch')
        left,top,right,bottom=max(nx0,ox0),max(ny0,oy0),min(nx1,ox1),min(ny1,oy1);transferred=0
        if left<right and top<bottom:
            win=Window(left-ox0,top-oy0,right-left,bottom-top);selected=old_roi.read(1,window=win)>0
            target=data[top-ny0:bottom-ny0,left-nx0:right-nx0];edits=old_src.read(1,window=win)
            transferred=int((selected&(target!=edits)).sum());target[selected]=edits[selected]
    output.mkdir(parents=True);save_labels(output/'region_result.tif',data,profile)
    inputs=[Path(plan_dir).resolve()/'plan.json',experiment/'experiment.json',score_dir/'scores.json',previous/'region_result.json',previous/'region_result.tif']
    workflow=dict(status='ready',experiment=str(experiment),fusion_scores=str(score_dir),start_method=method,target_id=p['instance_id'],
        previous_result=None,interaction_round=0,reason='Transferred approved edits into a separately evaluated region',operation='region-transfer',
        input_sha256={str(path):digest(path) for path in inputs})
    write_json(output/'workflow.json',workflow)
    report=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),interaction_project=str(output),start_method=method,
        reason=workflow['reason'],operation='region-transfer',write_policy='transferred-approved-edits',interaction_round=0,
        previous_result=None,transferred_from=str(previous),changed_pixels=transferred,ground_truth_used_for_edits=False,
        input_sha256={str(path):digest(path) for path in [output/'workflow.json',output/'region_result.tif',*inputs]})
    write_json(output/'region_result.json',report);print(f'Transferred {transferred} changed pixels in overlap to {output}; score this new-region starting result before SAM')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan');a.add_argument('--result-dir',type=Path,required=True);a.add_argument('--instance-id',type=int,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--plan-dir',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a=sub.add_parser('initialize');a.add_argument('--plan-dir',type=Path,required=True);a.add_argument('--experiment-dir',type=Path,required=True);a.add_argument('--fusion-scores',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='plan':plan(a.result_dir,a.instance_id,a.output_dir)
    elif a.command=='prepare':prepare(a.plan_dir,a.output_dir)
    else:initialize(a.plan_dir,a.experiment_dir,a.fusion_scores,a.output_dir)


if __name__=='__main__':main()
