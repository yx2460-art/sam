"""Reviewed, local GrabCut boundary refinement; no GT-driven prompt or edit selection."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .manual_fusion import read
from .metrics import label_boundary
from .raster import save_labels
from .refine_conflicts import save_mask
from .region_workflow import load,write_json
from .report_regions import overlay,save_png
from .run_tiles import digest
from .split_instances import sealed,rgb_read


def refine(rgb,source,roi,target_id,bounds,points,labels,width=8,iterations=5,cutter=None):
    if type(width) is not int or not 1<=width<=32 or type(iterations) is not int or not 1<=iterations<=10:raise ValueError('Invalid band width/iterations')
    if len(bounds)!=4 or any(type(v) is not int for v in bounds):raise ValueError('Integer rectangle xyxy required')
    x0,y0,x1,y1=bounds;h,w=source.shape
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Rectangle outside image')
    if not points or len(points)!=len(labels) or len(points)>32 or any(type(v) is not int or v not in (0,1) for v in labels) or set(labels)!={0,1}:raise ValueError('Provide both positive and negative points, maximum 32')
    target=source==target_id
    if target_id<=0 or not np.any(target&roi):raise ValueError('Target absent from fixed scoring ROI')
    rectangle=np.zeros_like(roi);rectangle[y0:y1,x0:x1]=True
    band=ndimage.maximum_filter(label_boundary(target.astype(np.uint8)),size=2*width+1,mode='constant')
    editable=band&rectangle&roi&((source==0)|target)
    if not editable.any():raise ValueError('Rectangle has no editable target boundary')
    gc=np.where(target,1,0).astype(np.uint8)
    gc[editable&target]=3;gc[editable&~target]=2
    seen=set()
    for p,label in zip(points,labels):
        if len(p)!=2 or any(type(v) is not int for v in p):raise ValueError('Integer points required')
        x,y=p
        if tuple(p) in seen or not 0<=x<w or not 0<=y<h or not editable[y,x]:raise ValueError('Point duplicated or outside editable boundary band/rectangle/ROI, or on protected instance')
        seen.add(tuple(p));gc[y,x]=label
    if min(int(np.isin(gc,[1,3]).sum()),int(np.isin(gc,[0,2]).sum()))<5:raise ValueError('Insufficient foreground/background samples')
    if cutter is None:
        import cv2
        cv2.setRNGSeed(0)
        try:cv2.grabCut(np.ascontiguousarray(rgb),gc,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),iterations,cv2.GC_INIT_WITH_MASK)
        except cv2.error as exc:raise ValueError('GrabCut could not fit this image; no forced boundary output') from exc
    else:cutter(rgb,gc)
    selected=np.isin(gc,[1,3]);after=source.copy()
    after[editable&selected]=target_id;after[editable&~selected]=0
    for (x,y),label in zip(points,labels):
        if bool(after[y,x]==target_id)!=bool(label):raise ValueError('Candidate violates an explicit prompt')
    return after,editable


def prepare(result,target_id,output,width=8):
    result=Path(result).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new project directory')
    if not 1<=width<=32:raise ValueError('Band width must be 1 to 32 pixels')
    prior=sealed(result/'region_result.json');old_project=Path(prior['interaction_project']);old=sealed(old_project/'workflow.json')
    experiment,m=load(old['experiment'])
    if prior['experiment_sha256']!=digest(experiment/'experiment.json'):raise ValueError('Region mismatch')
    source=result/'region_result.tif'
    with rasterio.open(source) as src,rasterio.open(m['mask']) as roi,rasterio.open(m['image']) as image:
        data=src.read(1);mask=roi.read(1)>0;rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
    target=data==target_id
    if target_id<=0 or not np.any(target&mask):raise ValueError('Positive target ID absent from ROI')
    band=ndimage.maximum_filter(label_boundary(target.astype(np.uint8)),size=2*width+1,mode='constant')&mask&((data==0)|target)
    output.mkdir(parents=True);save_png(output/'image.png',rgb)
    view=overlay(rgb,data,mask);view[band]=(.5*view[band]+.5*np.array([0,220,255])).astype(np.uint8);save_png(output/'band.png',view)
    runs=[]
    for row in band:
        changes=np.diff(np.r_[0,row.astype(np.int8),0]);runs.append([[int(a),int(b)] for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))])
    inputs=[source,result/'region_result.json',old_project/'workflow.json',experiment/'experiment.json',Path(m['image']),Path(m['mask'])]
    plan=dict(source_result=str(result),source=str(source),experiment=str(experiment),target_id=target_id,width=width,
        image=m['image'],mask=m['mask'],input_sha256={str(p):digest(p) for p in inputs})
    write_json(output/'plan.json',plan)
    workflow=dict(old,previous_result=str(result),target_id=target_id,reason='Human-scoped local GrabCut boundary refinement',operation='local-boundary-refinement',
        input_sha256={str(p):digest(p) for p in [output/'plan.json',old_project/'workflow.json',experiment/'experiment.json']})
    write_json(output/'workflow.json',workflow)
    config=json.dumps(dict(hash=digest(output/'plan.json'),allowed=runs))
    script='const c='+config+''',canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),img=new Image(),corners=[],points=[],labels=[],began=Date.now();
function draw(){ctx.drawImage(img,0,0);if(corners.length===2){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.strokeRect(corners[0][0],corners[0][1],corners[1][0]-corners[0][0],corners[1][1]-corners[0][1]);}points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fillStyle=labels[i]?'red':'blue';ctx.fill();});}
img.onload=draw;img.src='band.png';document.getElementById('raw').onchange=e=>img.src=e.target.checked?'image.png':'band.png';
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),p=[Math.floor((e.clientX-r.left)*canvas.width/r.width),Math.floor((e.clientY-r.top)*canvas.height/r.height)],tool=document.getElementById('tool').value;
if(tool==='box'){if(corners.length===2){alert('先清除矩形再重画');return;}corners.push(p);draw();return;}
if(corners.length!==2){alert('先用两个对角点框定局部误差位置');return;}if(points.length>=32){alert('最多32点');return;}
if(!c.allowed[p[1]]?.some(([a,b])=>p[0]>=a&&p[0]<b)){alert('请在青色可编辑边界带内标点，其他实例受保护');return;}
const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]);if(p[0]<Math.min(...xs)||p[0]>Math.max(...xs)||p[1]<Math.min(...ys)||p[1]>Math.max(...ys)){alert('请在局部矩形内标点');return;}points.push(p);labels.push(Number(tool));draw();};
function reset(){corners.length=0;points.length=0;labels.length=0;draw();}function undo(){points.pop();labels.pop();draw();}
function download(){if(corners.length!==2||!labels.includes(0)||!labels.includes(1)){alert('需要局部矩形以及正、负两类点');return;}
const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]),bounds=[Math.min(...xs),Math.min(...ys),Math.max(...xs)+1,Math.max(...ys)+1];
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({plan_sha256:c.hash,bounds,points,labels,browser_wall_seconds:(Date.now()-began)/1000},null,2)],{type:'application/json'}));a.download='boundary_points.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}'''
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>局部边界修正对照</h1><p>青色为可编辑边界带。先在误差附近点两个对角点画局部矩形，再在带内标红色正点（田块）和蓝色负点（田埂／背景）。只修一个明确位置，不按GT描边。其他实例及矩形／边界带之外均保护。</p><select id="tool"><option value="box">框局部矩形</option><option value="1">正点：田块</option><option value="0">负点：背景</option></select><button onclick="reset()">清空重画</button><button onclick="undo()">撤销点</button><label><input id="raw" type="checkbox">只看原图</label><button onclick="download()">导出boundary_points.json</button>'
        f'<p>目标ID {target_id}；边界带半宽 {width} 像素。此处矩形仅限定局部可修改范围，不是SAM斜框。</p><canvas id="map" width="{data.shape[1]}" height="{data.shape[0]}" style="max-width:97vw;height:auto"></canvas><script>'+script+'</script>',encoding='utf-8')
    print(f'Open {output}/index.html; no GT read; local graph cut, not SAM')
    return plan


def infer(project,points_path,output,cutter=None):
    began=perf_counter()
    project=Path(project).resolve();output=Path(output).resolve();p=sealed(project/'plan.json');a=read(points_path)
    if output.exists():raise FileExistsError('Use a new candidate directory')
    if a.get('plan_sha256')!=digest(project/'plan.json'):raise ValueError('Prompts belong to another boundary project')
    with rasterio.open(p['source']) as src,rasterio.open(p['image']) as image,rasterio.open(p['mask']) as roi:
        before=src.read(1);mask=roi.read(1)>0;rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
        after,editable=refine(rgb,before,mask,p['target_id'],a['bounds'],a['points'],a['labels'],p['width'],cutter=cutter)
        output.mkdir(parents=True);save_labels(output/'candidate.tif',after,src.profile);save_mask(output/'editable.tif',editable,src.profile)
    save_png(output/'before.png',overlay(rgb,before,mask));save_png(output/'after.png',overlay(rgb,after,mask))
    inputs=[project/'plan.json',project/'workflow.json',Path(points_path).resolve(),output/'candidate.tif',output/'editable.tif']
    report=dict(project=str(project),bounds=a['bounds'],points=a['points'],labels=a['labels'],browser_wall_seconds=a.get('browser_wall_seconds'),
        method='OpenCV GrabCut with mask initialization',band_width=p['width'],iterations=5,random_seed=0,
        compute_seconds=perf_counter()-began,
        target_components_before=int(ndimage.label((before==p['target_id'])&mask,structure=np.ones((3,3)))[1]),
        target_components_after=int(ndimage.label((after==p['target_id'])&mask,structure=np.ones((3,3)))[1]),
        changed_pixels=int((before!=after).sum()),added_foreground_pixels=int(((before==0)&(after>0)).sum()),removed_foreground_pixels=int(((before>0)&(after==0)).sum()),
        input_sha256={str(path):digest(path) for path in inputs})
    write_json(output/'proposal.json',report)
    script='function download(){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({proposal_sha256:"'+digest(output/'proposal.json')+'",accepted:document.getElementById("accept").checked},null,2)],{type:"application/json"}));a.download="boundary_approval.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}'
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>局部图割候选确认</h1><p>左：之前；右：候选。黄色为边界，无GT叠加。检查是否贴合原影像田埂、有无断裂；默认不接受。此操作可能增加或删除前景。</p><img src="before.png" style="width:48%"><img src="after.png" style="width:48%">'
        f'<p>改变 {report["changed_pixels"]} 像素，新增前景 {report["added_foreground_pixels"]}，删除前景 {report["removed_foreground_pixels"]}。目标在评分域内的8连通区域数量：{report["target_components_before"]} → {report["target_components_after"]}（变化不等同于正确性，注意是否断裂）。</p><label><input type="checkbox" id="accept">接受局部边界修改</label><button onclick="download()">导出boundary_approval.json</button><script>'+script+'</script>',encoding='utf-8')
    print(f'Open {output}/review.html; changed pixels={report["changed_pixels"]}; no GT used')
    return report


def apply(proposals,approval,output):
    proposals=Path(proposals).resolve();output=Path(output).resolve();r=sealed(proposals/'proposal.json');project=Path(r['project']);p=sealed(project/'plan.json');w=sealed(project/'workflow.json');a=read(approval)
    if output.exists():raise FileExistsError('Use a new result directory')
    if a.get('proposal_sha256')!=digest(proposals/'proposal.json') or a.get('accepted') is not True:raise ValueError('Candidate not explicitly accepted or approval mismatch')
    with rasterio.open(p['source']) as src,rasterio.open(proposals/'candidate.tif') as candidate,rasterio.open(proposals/'editable.tif') as edit:
        before=src.read(1);after=candidate.read(1);editable=edit.read(1)>0
        if after.shape!=before.shape or not candidate.transform.almost_equals(src.transform):raise ValueError('Candidate grid mismatch')
        changed=before!=after
        if np.any(changed&~editable) or np.any(changed&~np.isin(after,[0,p['target_id']])):raise ValueError('Candidate outside permitted edit scope')
        output.mkdir(parents=True);save_labels(output/'region_result.tif',after,src.profile);save_mask(output/'changed_pixels.tif',changed,src.profile)
    prior=sealed(Path(p['source_result'])/'region_result.json');experiment,m=load(p['experiment'])
    inputs=[project/'workflow.json',proposals/'proposal.json',Path(approval).resolve(),output/'region_result.tif']
    report=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),interaction_project=str(project),start_method=prior['start_method'],
        reason=w['reason'],previous_result=p['source_result'],operation='local-grabcut-boundary',write_policy='reviewed-local-boundary',interaction_round=prior.get('interaction_round',1),
        changed_pixels=int(changed.sum()),added_foreground_pixels=r['added_foreground_pixels'],removed_foreground_pixels=r['removed_foreground_pixels'],
        prompt_points=len(r['points']),rectangle_clicks=2,browser_wall_seconds=r.get('browser_wall_seconds'),ground_truth_used_for_edits=False,
        compute_seconds=r['compute_seconds'],
        input_sha256={str(path):digest(path) for path in inputs})
    write_json(output/'region_result.json',report);print(f'Exported {output}/region_result.tif; local boundary control only')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--target-id',type=int,required=True);p.add_argument('--band-width',type=int,default=8);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('infer');p.add_argument('--project-dir',type=Path,required=True);p.add_argument('--points',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--proposals-dir',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.result_dir,a.target_id,a.output_dir,a.band_width)
    elif a.command=='infer':infer(a.project_dir,a.points,a.output_dir)
    else:apply(a.proposals_dir,a.approval,a.output_dir)


if __name__=='__main__':main()
