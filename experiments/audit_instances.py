"""Post-hoc instance matching audit on the existing development ROI."""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .metrics import instance_counts
from .run_tiles import aligned, digest


def analyze(pred, gt, mask, origin=(0,0), evidence_fraction=0.1):
    if pred.shape!=gt.shape or pred.shape!=mask.shape or pred.ndim!=2:
        raise ValueError('Expected matching two-dimensional label/mask grids')
    for data in (pred,gt):
        if not np.issubdtype(data.dtype,np.integer) or np.any(data<0):
            raise ValueError('Expected nonnegative integer instance labels')
    if not 0 < evidence_fraction <= 1:
        raise ValueError('Evidence fraction must be in (0,1]')
    p,t=np.where(mask,pred,0),np.where(mask,gt,0)
    pa=dict(zip(*np.unique(p[p>0],return_counts=True)))
    ga=dict(zip(*np.unique(t[t>0],return_counts=True)))
    shared=(p>0)&(t>0)
    pairs,counts=np.unique(np.stack([p[shared],t[shared]],axis=1),axis=0,return_counts=True)
    overlaps={}
    candidates=[]
    for (pid,gid),count in zip(pairs,counts):
        pid,gid,count=int(pid),int(gid),int(count)
        iou=count/(int(pa[pid])+int(ga[gid])-count)
        overlaps[(pid,gid)]=(count,iou)
        if iou>=0.5:
            candidates.append((iou,pid,gid))
    used_p,used_g,matched=set(),set(),{}
    for iou,pid,gid in sorted(candidates,reverse=True):
        if pid not in used_p and gid not in used_g:
            used_p.add(pid)
            used_g.add(gid)
            matched[pid]=gid
    # Two or more predictions covering >=10% each of one GT suggest fragmentation;
    # one prediction covering >=10% of each of multiple GTs suggests merging.
    links={(pid,gid) for (pid,gid),(count,_) in overlaps.items()
           if count/int(ga[gid])>=evidence_fraction}
    merge={pid for pid in pa if sum(a==pid for a,b in links)>=2}
    split={gid for gid in ga if sum(b==gid for a,b in links)>=2}
    rows=[]
    for kind,areas,data,full in [('prediction',pa,p,pred),('gt',ga,t,gt)]:
        edge_ids=set(np.unique(np.concatenate([full[0],full[-1],full[:,0],full[:,-1]])))
        clipped=set(np.unique(full[~mask]))|edge_ids
        for ident,area in sorted(areas.items()):
            ident=int(ident)
            choices=[(iou,pid,gid,count) for (pid,gid),(count,iou) in overlaps.items()
                     if (pid if kind=='prediction' else gid)==ident]
            best=max(choices,default=None)
            partner=matched.get(ident) if kind=='prediction' else next((pid for pid,gid in matched.items() if gid==ident),None)
            y,x=np.nonzero(data==ident)
            merge_flag=ident in merge if kind=='prediction' else any((pid,ident) in links for pid in merge)
            split_flag=ident in split if kind=='gt' else any((ident,gid) in links for gid in split)
            reason='matched' if partner is not None else ('no_foreground_overlap' if best is None else
                   ('below_iou_0.5' if best[0]<0.5 else 'one_to_one_competition'))
            rows.append(dict(kind=kind,id=ident,area_pixels=int(area),matched_id=partner,
                status=reason,best_iou=best[0] if best else 0.0,
                best_partner=(best[2] if kind=='prediction' else best[1]) if best else None,
                merge_evidence=bool(merge_flag),split_evidence=bool(split_flag),
                roi_or_crop_cut_possible=ident in clipped,
                x_min=int(x.min())+origin[0],y_min=int(y.min())+origin[1],
                x_max_exclusive=int(x.max())+origin[0]+1,y_max_exclusive=int(y.max())+origin[1]+1))
    summary=dict(instance_counts(p,t),prediction_count=len(pa),gt_count=len(ga),
                 predictions_with_merge_evidence=len(merge),gt_with_split_evidence=len(split))
    for kind in ('prediction','gt'):
        subset=[r for r in rows if r['kind']==kind]
        summary[kind+'_unmatched_by_reason']={reason:sum(r['status']==reason for r in subset)
             for reason in ('no_foreground_overlap','below_iou_0.5','one_to_one_competition')}
        summary[kind+'_possible_roi_or_crop_cut']=sum(r['roi_or_crop_cut_possible'] for r in subset)
    return summary,rows


def run(boundary_dir,gt_path,output_dir):
    boundary_dir,gt_path,output_dir=map(lambda p:Path(p).resolve(),(boundary_dir,gt_path,output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Use a new directory: {output_dir}')
    report_path=boundary_dir/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status')!='complete' or 'evaluation' not in report:
        raise ValueError('Expected completed evaluated boundary replay')
    for path,expected in report['input_sha256'].items():
        if digest(path)!=expected:
            raise ValueError(f'Changed source: {path}')
    region_dir=Path(report['source_run'])
    original=json.loads((region_dir/'report.json').read_text(encoding='utf-8'))
    union_dir=Path(original['source_union'])
    manifest=json.loads((union_dir/'manifest.json').read_text(encoding='utf-8'))
    for parent in (boundary_dir,region_dir,union_dir,Path(manifest['source_run'])):
        if output_dir==parent.resolve() or parent.resolve() in output_dir.parents:
            raise ValueError('Output must be outside source runs')
    if digest(gt_path)!=original.get('gt_sha256'):
        raise ValueError('GT differs from original evaluation')
    params=manifest['parameters']
    paths={'bsinet':Path(params['prediction']),'union':union_dir/'instance.tif',
           'conflict_unrestricted':region_dir/'conflict'/'instance.tif',
           'random_unrestricted':region_dir/'random'/'instance.tif',
           'conflict_boundary':boundary_dir/'conflict'/'instance.tif',
           'random_boundary':boundary_dir/'random'/'instance.tif'}
    inputs=[report_path,region_dir/'report.json',union_dir/'manifest.json',gt_path,
            Path(params['eval_mask']),*paths.values()]
    hashes={str(p):digest(p) for p in inputs}
    x0,y0,x1,y1=manifest['crop_xyxy']
    window=Window(x0,y0,x1-x0,y1-y0)
    summaries={}
    rows=[]
    with ExitStack() as stack:
        base=stack.enter_context(rasterio.open(paths['bsinet']))
        union=stack.enter_context(rasterio.open(paths['union']))
        gt=stack.enter_context(rasterio.open(gt_path))
        roi=stack.enter_context(rasterio.open(params['eval_mask']))
        allow=params.get('allow_crs_mismatch',False)
        aligned(base,gt,allow)
        aligned(base,roi,allow)
        if union.shape!=(y1-y0,x1-x0) or not union.transform.almost_equals(base.window_transform(window)):
            raise ValueError('Union crop grid mismatch')
        raw=roi.read(1,window=window)
        mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=window)>0)
        if not mask.any():
            raise ValueError('Empty scoring ROI')
        target=gt.read(1,window=window)
        for name,path in paths.items():
            src=stack.enter_context(rasterio.open(path))
            if name!='bsinet':
                aligned(union,src,allow)
            pred=src.read(1,window=window) if name=='bsinet' else src.read(1)
            summary,items=analyze(pred,target,mask,(x0,y0))
            expected=report['evaluation'][name]['counts']
            if any(summary[k]!=expected[k] for k in ('instance_tp','instance_fp','instance_fn','matched_iou_sum')):
                raise ValueError(f'{name}: instance counts differ from saved evaluation')
            summaries[name]=summary
            rows.extend(dict(method=name,**r) for r in items)
    if any(digest(p)!=h for p,h in hashes.items()):
        raise ValueError('Source changed during audit')
    output_dir.mkdir(parents=True)
    with (output_dir/'instances.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=list(rows[0]) if rows else ['method','kind','id','area_pixels','matched_id','status',
            'best_iou','best_partner','merge_evidence','split_evidence','roi_or_crop_cut_possible',
            'x_min','y_min','x_max_exclusive','y_max_exclusive']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result=dict(scope='Development ROI-clipped IDs; no new inference; training independence unverified',
        matching='Same greedy descending IoU >=0.5 one-to-one matching as existing metrics',
        evidence='Merge/split flags are overlapping heuristics, not confirmed causes: overlap >=10% of GT area; includes matched objects',
        crop_warning='Cut flag means same ID outside ROI in cached crop or touching crop edge; absence does not certify a complete parcel',
        input_sha256=hashes,methods=summaries)
    (output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=[result['scope'],result['evidence']]
    for name,s in summaries.items():
        lines.append(f"{name}: TP={s['instance_tp']}, FP={s['instance_fp']}, FN={s['instance_fn']}; "
                     f"unmatched_GT={s['gt_unmatched_by_reason']}; merge_predictions={s['predictions_with_merge_evidence']}, split_GT={s['gt_with_split_evidence']}")
    text='\n'.join(lines)+'\n'
    (output_dir/'summary.txt').write_text(text,encoding='utf-8')
    print(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.boundary_dir,args.gt,args.output_dir)


if __name__=='__main__':
    main()
