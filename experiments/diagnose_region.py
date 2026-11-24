"""Post-evaluation regional boundary and instance diagnostics; never edits predictions."""
import argparse
import csv
import html
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .metrics import label_boundary
from .raster import save_labels
from .region_workflow import load,fusion_scores,write_json
from .report_regions import save_png
from .run_tiles import digest
from .split_instances import sealed,rgb_read


def boundary_errors(pred,gt,mask,tolerance):
    """Exactly the same erosion and square tolerance as roi_counts."""
    interior=ndimage.binary_erosion(mask,iterations=tolerance+1,border_value=0)
    pb,gb=label_boundary(pred),label_boundary(gt);size=2*tolerance+1
    matched=pb&interior&ndimage.maximum_filter(gb,size=size,mode='constant')
    extra=pb&interior&~ndimage.maximum_filter(gb,size=size,mode='constant')
    missing=gb&interior&~ndimage.maximum_filter(pb,size=size,mode='constant')
    codes=np.zeros(mask.shape,dtype=np.uint32);codes[matched]=1;codes[extra]=2;codes[missing]=3
    return codes,dict(boundary_pred=int((pb&interior).sum()),boundary_gt=int((gb&interior).sum()),
        boundary_matched_pred=int(matched.sum()),boundary_matched_gt=int((gb&interior).sum()-missing.sum()))


def instance_matches(pred,gt,mask):
    """Expose the same descending-IoU greedy matches used by the evaluator."""
    p=np.where(mask,pred,0);g=np.where(mask,gt,0)
    ids,counts=np.unique(p[p>0],return_counts=True);areas=dict(zip(ids.tolist(),counts.tolist()))
    gi,gc=np.unique(g[g>0],return_counts=True);ga=dict(zip(gi.tolist(),gc.tolist()))
    overlap=(p>0)&(g>0)
    pairs,counts=np.unique(np.stack([p[overlap],g[overlap]],axis=1),axis=0,return_counts=True)
    candidates=[];best={}
    for (pid,gid),count in zip(pairs,counts):
        pid,gid=int(pid),int(gid);iou=float(count/(areas[pid]+ga[gid]-count))
        if (iou,gid)>best.get(pid,(-1,0)):best[pid]=(iou,gid)
        if iou>=.5:candidates.append((iou,pid,gid))
    used_p={};used_g=set()
    for iou,pid,gid in sorted(candidates,reverse=True):
        if pid not in used_p and gid not in used_g:used_p[pid]=(gid,iou);used_g.add(gid)
    rows=[dict(pred_id=pid,pixels=area,best_gt=best.get(pid,(0,0))[1],best_iou=best.get(pid,(0,0))[0],
        matched_gt=used_p.get(pid,(None,0))[0],matched_iou=used_p.get(pid,(None,0))[1]) for pid,area in sorted(areas.items())]
    return rows,sorted(set(ga)-used_g)


def changes(before,after,gt,mask):
    added=mask&(before==0)&(after>0);removed=mask&(before>0)&(after==0)
    ids=mask&(before>0)&(after>0)&(before!=after)
    groups=[added&(gt>0),added&(gt==0),removed&(gt>0),removed&(gt==0),ids]
    names=['correct_added_foreground','wrong_added_foreground','wrong_removed_foreground','correct_removed_foreground','instance_id_changes']
    codes=np.zeros(mask.shape,dtype=np.uint32)
    for index,selected in enumerate(groups,1):codes[selected]=index
    return codes,dict(zip(names,[int(selected.sum()) for selected in groups]))


def paint(rgb,codes,mask,palette):
    view=rgb.copy();view[~mask]=(view[~mask]*.25).astype(np.uint8)
    for code,color in palette.items():view[codes==code]=color
    return view


def diagnose(experiment,result,gt_path,output):
    experiment,m=load(experiment);result=Path(result).resolve();output=Path(output).resolve();gt_path=Path(gt_path).resolve()
    if output.exists():raise FileExistsError('Use a new diagnostic output directory')
    # Reuse all score provenance/GT/grid checks; no SAM inference and no candidate selection.
    score=fusion_scores(experiment,gt_path,output/'scores',result)
    r=sealed(result/'region_result.json');workflow=sealed(Path(r['interaction_project'])/'workflow.json')
    paths={r['start_method']:Path(m['candidates'][r['start_method']])}
    if workflow.get('previous_result'):paths['previous_interactive']=Path(workflow['previous_result'])/'region_result.tif'
    paths['interactive']=result/'region_result.tif'
    x0,y0,x1,y1=m['context_xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
    with rasterio.open(m['image']) as image,rasterio.open(m['mask']) as roi,rasterio.open(gt_path) as gt:
        rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None);mask=roi.read(1)>0
        truth=gt.read(1,window=win);profile=roi.profile
    save_png(output/'rgb.png',rgb);arrays={};details={};cards=[]
    boundary_palette={1:[30,220,80],2:[255,40,40],3:[40,140,255]}
    for name,path in paths.items():
        with rasterio.open(path) as src:data=src.read(1)
        arrays[name]=data
        codes,counts=boundary_errors(data,truth,mask,m['boundary_tolerance'])
        expected=score['evaluation'][name]['counts']
        if any(expected[key]!=value for key,value in counts.items()):raise ValueError('Boundary diagnostic differs from evaluation')
        save_labels(output/f'{name}_boundary_errors.tif',codes,profile)
        save_png(output/f'{name}_boundary_errors.png',paint(rgb,codes,mask,boundary_palette))
        overlay=np.zeros(mask.shape,dtype=np.uint32);pb=label_boundary(data)&mask;gb=label_boundary(truth)&mask
        overlay[pb]=1;overlay[gb]=2;overlay[pb&gb]=3
        save_png(output/f'{name}_vs_gt.png',paint(rgb,overlay,mask,{1:[255,220,0],2:[0,230,255],3:[255,255,255]}))
        rows,missed=instance_matches(data,truth,mask)
        if sum(row['matched_gt'] is not None for row in rows)!=expected['instance_tp']:raise ValueError('Instance diagnostic differs from evaluation')
        with (output/f'{name}_instances.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            columns=['pred_id','pixels','best_gt','best_iou','matched_gt','matched_iou']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        details[name]=dict(boundary_counts=counts,unmatched_gt_ids=missed,instances=rows)
        cards.append(f'<h2>{html.escape(name)}</h2><p>预测边界黄／真值青／重合白；右图：容差内预测边界绿／容差外预测边界红／未匹配真值边界蓝。</p>'
            f'<div class="pair"><img src="{name}_vs_gt.png"><img src="{name}_boundary_errors.png"></div>'
            f'<p>预测边界像素 {counts["boundary_pred"]}；未匹配预测 {counts["boundary_pred"]-counts["boundary_matched_pred"]}；未匹配真值 {counts["boundary_gt"]-counts["boundary_matched_gt"]}。'
            f'<a href="{name}_instances.csv">逐实例匹配CSV</a>；未匹配真值ID：{missed}</p>')
    previous_name='previous_interactive' if 'previous_interactive' in arrays else r['start_method']
    codes,change_counts=changes(arrays[previous_name],arrays['interactive'],truth,mask)
    save_labels(output/'last_round_changes.tif',codes,profile)
    save_png(output/'last_round_changes.png',paint(rgb,codes,mask,{1:[20,220,80],2:[255,40,40],3:[40,140,255],4:[220,80,220],5:[255,220,0]}))
    new_ids=sorted(set(np.unique(arrays['interactive'][mask]).tolist())-set(np.unique(arrays[previous_name][mask]).tolist())-{0})
    report=dict(status='complete',warning='GT-visible development diagnosis; not a blind test and not an editing/prompt page',
        evaluation=score['evaluation'],boundary_tolerance=m['boundary_tolerance'],scoring_pixels=m['scoring_pixels'],
        context_xyxy=m['context_xyxy'],last_round_reference=previous_name,new_pred_ids=new_ids,
        last_round_changes=change_counts,diagnostics=details,
        input_sha256={str(p):digest(p) for p in [experiment/'experiment.json',result/'region_result.json',gt_path,*paths.values()]})
    write_json(output/'diagnostics.json',report)
    lines=['GT-visible diagnosis only; no predictions changed.',f'Scoring pixels={m["scoring_pixels"]}; tolerance={m["boundary_tolerance"]} pixels']
    for name,entry in details.items():lines.append(f'{name}: {entry["boundary_counts"]}')
    lines.extend([f'New instance IDs relative to {previous_name}: {new_ids}',f'Last-round changes: {change_counts}'])
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>区域交互后边界诊断</title>'
        '<style>body{font-family:sans-serif;margin:20px}.pair{display:flex;gap:1%}.pair img{width:49%;object-fit:contain}img{max-width:100%}p{line-height:1.6}</style>'
        '<h1>固定区域：边界与实例匹配诊断</h1><p>本页显示GT，仅供事后开发诊断，不生成提示，不自动修边，不替代原有评分。之后修改属于看过GT后的开发实验。'
        f'固定评分像素 {m["scoring_pixels"]}；边界容差 {m["boundary_tolerance"]} 像素，评分域向内收缩容差＋1，忽略人为ROI边缘。图片可在新标签页放大查看。</p>'
        '<p><a href="scores/summary.txt">相同设置的完整评分</a>；<a href="diagnostics.json">诊断数据</a></p>'
        +''.join(cards)+'<h2>本轮相对上一结果的改动</h2><p>绿：正确补前景；红：错误补前景；蓝：错误删前景；紫：正确删前景；黄：前景内ID改变（不等同于正确拆分）。</p>'
        '<img src="last_round_changes.png">'+f'<p>{html.escape(str(change_counts))}。新出现预测ID：{new_ids}。</p>',encoding='utf-8')
    print('\n'.join(lines));print(f'Open {output}/index.html; diagnosis only, no SAM inference')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True);parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();diagnose(a.experiment_dir,a.result_dir,a.gt,a.output_dir)


if __name__=='__main__':main()
