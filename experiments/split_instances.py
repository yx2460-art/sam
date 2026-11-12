"""Human-positive SAM instance refinement and background completion, with explicit approval."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .manual_fusion import read
from .raster import save_labels
from .report_regions import overlay,save_png
from .run_tiles import aligned,digest


def sealed(path):
    m=read(path)
    for name,value in m['input_sha256'].items():
        if digest(name)!=value: raise ValueError(f'Changed input: {name}')
    return m


def rgb_read(src,win,bands,value_range):
    rgb=src.read(bands,window=win)
    valid=np.all(src.read_masks(bands,window=win)>0,axis=0)&np.isfinite(rgb).all(axis=0)
    rgb=np.where(valid[None],rgb,0)
    if value_range:
        lo,hi=value_range
        if hi<=lo: raise ValueError('Invalid RGB range')
        rgb=np.clip((rgb.astype(float)-lo)*255/(hi-lo),0,255)
    elif rgb.dtype!=np.uint8: raise ValueError('Non-UInt8 image requires --rgb-range')
    return np.moveaxis(rgb.astype(np.uint8),0,-1)


def prepare(image_path,instances,output,ids=None,roi_path=None,pad=128,max_side=4096,
            bands=(1,2,3),value_range=None,allow=False):
    image_path,instances,output=map(lambda p:Path(p).resolve(),(image_path,instances,output))
    if output.exists(): raise FileExistsError('Use a new output directory')
    if pad<0 or max_side<=0: raise ValueError('Invalid context limits')
    boxes={};max_id=0
    with rasterio.open(image_path) as image,rasterio.open(instances) as src:
        aligned(image,src,allow)
        if not np.issubdtype(np.dtype(src.dtypes[0]),np.integer) or src.nodata not in (None,0):
            raise ValueError('Expected nonnegative integer instances, NoData=None/0')
        chosen=set(ids or [])
        if any(type(i) is not int or i<=0 for i in chosen): raise ValueError('Source instance IDs must be positive integers')
        if not chosen and roi_path:
            with rasterio.open(roi_path) as roi:
                aligned(image,roi,allow)
                for _,win in src.block_windows(1):
                    raw=roi.read(1,window=win);mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
                    chosen.update(int(i) for i in np.unique(src.read(1,window=win)[mask]) if i>0)
        if not chosen: raise ValueError('Specify --instance-ids or a nonempty --eval-mask for location selection')
        for _,win in src.block_windows(1):
            a=src.read(1,window=win)
            if np.any(a<0): raise ValueError('Negative instance IDs')
            max_id=max(max_id,int(a.max()))
            for lab in chosen.intersection(int(i) for i in np.unique(a)):
                y,x=np.where(a==lab);box=[int(x.min()+win.col_off),int(y.min()+win.row_off),
                                         int(x.max()+win.col_off+1),int(y.max()+win.row_off+1)]
                old=boxes.get(lab,box);boxes[lab]=[min(old[0],box[0]),min(old[1],box[1]),max(old[2],box[2]),max(old[3],box[3])]
        if chosen!=set(boxes): raise ValueError('Some requested IDs are absent')
        if max_id>=np.iinfo(np.uint32).max: raise ValueError('No UInt32 ID space available')
        plans=[]
        for lab,box in sorted(boxes.items()):
            x0,y0,x1,y1=box;x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(src.width,x1+pad);y1=min(src.height,y1+pad)
            if max(x1-x0,y1-y0)>max_side:
                raise ValueError(f'Instance {lab}: full context exceeds {max_side}; no silent truncation. Select fewer IDs or explicitly increase --max-side')
            plans.append(dict(id=lab,xyxy=[x0,y0,x1,y1]))
        output.mkdir(parents=True)
        cards=[];ownership={}
        for row in plans:
            lab=row['id'];x0,y0,x1,y1=row['xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
            original=src.read(1,window=win);target=original==lab;rgb=rgb_read(image,win,bands,value_range)
            ownership[lab]=[]
            for line in original:
                starts=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1]
                ownership[lab].append([[int(x),int(line[x])] for x in starts])
            if np.any(target&~np.all(image.read_masks(bands,window=win)>0,axis=0)):
                raise ValueError(f'Instance {lab} includes invalid image pixels')
            save_png(output/f'{lab}_rgb.png',rgb)
            save_png(output/f'{lab}_target.png',overlay(rgb,target.astype(np.uint32),np.ones_like(target)))
            row['source_pixels']=int(target.sum())
            cards.append(f'<section><h2>实例 {lab} · 原图左上角 ({x0},{y0})</h2>'
                         f'<p>黄色为原预测实例边界，不是标点限制。每个不同田块点一次，1至8点。背景点可补充漏检；已有其他ID会提示并保护。</p>'
                         f'<canvas id="c{lab}" width="{x1-x0}" height="{y1-y0}"></canvas>'
                         f'<p><button onclick="undo({lab})">撤销最后一点</button><button onclick="clearPoints({lab})">清空</button>'
                         f'<span id="s{lab}"></span></p></section>')
    inputs=[image_path,instances]+([Path(roi_path).resolve()] if roi_path else [])
    m=dict(status='ready',image=str(image_path),instances=str(instances),regions=plans,max_source_id=max_id,
           rgb_bands=list(bands),rgb_range=value_range,allow_crs_mismatch=allow,
           input_sha256={str(p):digest(p) for p in inputs},
           protocol='One positive per parcel; parent/background candidates; other existing IDs protected; no GT; residual preserved')
    (output/'project.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    config=json.dumps(dict(hash=digest(output/'project.json'),regions=plans,ownership=ownership))
    script=r'''
const config=CONFIG,points=Object.fromEntries(config.regions.map(r=>[r.id,[]])),images={},events=[],started=Date.now();
function draw(id){const c=document.getElementById('c'+id),ctx=c.getContext('2d');ctx.drawImage(images[id],0,0);
points[id].forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.fillStyle='#ff3030';ctx.fill();ctx.fillStyle='white';ctx.font='18px sans-serif';ctx.fillText(i+1,p[0]+8,p[1]);});
document.getElementById('s'+id).textContent=points[id].length+' 个田块正点。'+points[id].map((p,i)=>{
const rows=config.ownership[id],line=rows[p[1]];if(!line||p[0]<0||p[0]>=c.width)return '点'+(i+1)+'超出影像';
let label=line[0][1];for(const run of line){if(run[0]>p[0])break;label=run[1];}
return '点'+(i+1)+'：'+(label===id?'原实例内':label===0?'预测背景，可补充候选':'已有实例ID '+label+'，保持保护，请单独打开该ID');}).join('；');}
function undo(id){points[id].pop();events.push({id,action:'undo',utc:new Date().toISOString()});draw(id);}function clearPoints(id){points[id]=[];events.push({id,action:'clear',utc:new Date().toISOString()});draw(id);}
function download(){const rows=config.regions.filter(r=>points[r.id].length).map(r=>({id:r.id,points:points[r.id]}));
if(rows.some(r=>r.points.length<1||r.points.length>8)){alert('每个区域需要1至8个不同田块正点');return;}
if(!rows.length){alert('请先标点');return;}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({project_sha256:config.hash,instances:rows,events,browser_wall_seconds:(Date.now()-started)/1000},null,2)],{type:'application/json'}));a.download='split_points.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
config.regions.forEach(r=>{const id=r.id,c=document.getElementById('c'+id),im=new Image();images[id]=im;im.onload=()=>draw(id);im.src=id+'_target.png';
c.onclick=e=>{const box=c.getBoundingClientRect();points[id].push([Math.floor((e.clientX-box.left)*c.width/box.width),Math.floor((e.clientY-box.top)*c.height/box.height)]);events.push({id,action:'point',utc:new Date().toISOString()});draw(id);};});
'''.replace('CONFIG',config)
    page='<!doctype html><meta charset="utf-8"><title>实例拆分正点标注</title><style>body{font-family:sans-serif}canvas{max-width:95vw;height:auto;border:1px solid #888}header{position:sticky;top:0;background:white;padding:12px;z-index:1}</style><header><b>完整实例正点拆分</b> 每个不同田块点一次；不看GT。<button onclick="download()">导出 split_points.json</button></header>'+''.join(cards)+'<script>'+script+'</script>'
    (output/'index.html').write_text(page,encoding='utf-8');print(f'Open {output}/index.html; no GT read')
    return m


def split_candidate(rgb,parent,points,predictor,min_area=30,min_score=.8,source=None,valid=None,diagnostic_sink=None,prompt_mode='point-box'):
    """Reject ambiguous masks; never manufacture a geometric split if SAM cannot separate seeds."""
    flexible=source is not None
    if prompt_mode not in ('point-box','point-only'): raise ValueError('Unknown prompt mode')
    if not (1 if flexible else 2)<=len(points)<=8: raise ValueError('Use 1 to 8 seeds (strict splitting requires 2)')
    if valid is None: valid=np.ones_like(parent)
    seeds=[]
    for p in points:
        if len(p)!=2 or any(type(v) is not int for v in p): raise ValueError('Integer pixel points required')
        x,y=p
        if not (0<=y<parent.shape[0] and 0<=x<parent.shape[1] and valid[y,x]):
            raise ValueError('Positive point outside valid image context')
        if not flexible and not parent[y,x]: raise ValueError('Positive point outside selected instance')
        seeds.append((x,y))
    if len(set(seeds))!=len(seeds): raise ValueError('Duplicate seeds')
    predictor.set_image(rgb);ys,xs=np.where(parent)
    box=np.array([xs.min(),ys.min(),xs.max(),ys.max()],dtype=np.float32)
    logits=[];details=[];rejected=[];diagnostics=[]
    support=(parent|(source==0))&valid if flexible else parent
    for index,(x,y) in enumerate(seeds):
        prompt_box=None if prompt_mode=='point-only' or (flexible and source[y,x]==0) else box
        seed_record=dict(seed=index,point=[x,y],candidates=[],status='pending',
                         prompt_mode=prompt_mode,box=None if prompt_box is None else prompt_box.tolist())
        diagnostics.append(seed_record)
        if flexible and source[y,x]>0 and not parent[y,x]:
            seed_record.update(status='protected',source_id=int(source[y,x]))
            rejected.append(dict(seed=index,source_id=int(source[y,x]),reason='Point belongs to another existing instance; protected, open that instance separately'))
            continue
        masks,quality,_=predictor.predict(point_coords=np.array([[x,y]],dtype=np.float32),
            point_labels=np.ones(1,dtype=np.int32),box=prompt_box,
            multimask_output=True,return_logits=True)
        candidates=[]
        for candidate_index,(raw,score) in enumerate(zip(masks,quality)):
            finite=bool(np.isfinite(raw).all());finite_score=bool(np.isfinite(score))
            raw_mask=(raw>predictor.model.mask_threshold)&np.isfinite(raw)
            binary=raw_mask&support
            cc,_=ndimage.label(binary,structure=np.ones((3,3)))
            contains_own=bool(cc[y,x]!=0)
            component=cc==cc[y,x] if contains_own else np.zeros_like(parent)
            other=[j+1 for j,(xx,yy) in enumerate(seeds) if j!=index and component[yy,xx]]
            reasons=[]
            if not finite_score: reasons.append('non_finite_score')
            elif score<min_score: reasons.append('low_score')
            if not finite: reasons.append('non_finite_mask')
            if not contains_own: reasons.append('own_point_missing')
            if other: reasons.append('contains_other_points')
            if contains_own and component.sum()<min_area: reasons.append('too_small')
            record=dict(seed=index,candidate=candidate_index,score=float(score) if finite_score else None,
                min_score=min_score,min_area=min_area,mask_threshold=float(predictor.model.mask_threshold),
                raw_pixels=int(raw_mask.sum()),supported_pixels=int(binary.sum()),component_pixels=int(component.sum()),
                clipped_pixels=int((raw_mask&~support).sum()),own_point_in_raw=bool(raw_mask[y,x]),
                own_point_in_component=contains_own,
                raw_contains_points=[j+1 for j,(xx,yy) in enumerate(seeds) if raw_mask[yy,xx]],
                component_contains_other_points=other,
                touches_context_edge=bool(component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any()),
                rejection_codes=reasons,eligible=not reasons,selected=False)
            seed_record['candidates'].append(record)
            if diagnostic_sink: diagnostic_sink(record,raw_mask,binary,component)
            if reasons: continue
            candidates.append((float(score),raw,component,candidate_index))
        if not candidates:
            seed_record['status']='no_eligible_candidate'
            if not flexible: return None,dict(reason='SAM cannot separate at least one seed from other child seeds',seed=index,diagnostics=diagnostics)
            rejected.append(dict(seed=index,reason='SAM cannot distinguish this point or no candidate meets quality/area thresholds'))
            continue
        score,raw,component,candidate_index=max(candidates,key=lambda r:r[0])
        seed_record.update(status='selected_before_overlap',selected_candidate=candidate_index)
        seed_record['candidates'][candidate_index]['selected']=True
        logits.append(np.where(component,raw,-np.inf));details.append(dict(seed=index,score=score,pixels=int(component.sum()),
            candidate=candidate_index,kind='background_completion' if flexible and source[y,x]==0 else 'parent_refinement'))
    if not logits: return None,dict(reason='No separable candidate',rejected_seeds=rejected,diagnostics=diagnostics)
    values=np.stack(logits);covered=np.isfinite(values).any(axis=0)
    indices=np.array([d['seed']+1 for d in details],dtype=np.uint32)
    assigned=np.where(covered,indices[np.argmax(values,axis=0)],0).astype(np.uint32)
    kept=[]
    for detail in details:
        index=detail['seed']+1;x,y=seeds[index-1]
        cc,_=ndimage.label(assigned==index,structure=np.ones((3,3)))
        if cc[y,x]==0:
            diagnostics[index-1]['status']='lost_seed_after_overlap'
            assigned[assigned==index]=0;rejected.append(dict(seed=index-1,reason='Seed lost during overlap resolution'));continue
        component=cc==cc[y,x];assigned[(assigned==index)&~component]=0
        if component.sum()<min_area:
            diagnostics[index-1]['status']='too_small_after_overlap'
            assigned[assigned==index]=0;rejected.append(dict(seed=index-1,reason='Child too small after overlap resolution'));continue
        diagnostics[index-1].update(status='retained',final_pixels=int(component.sum()))
        kept.append(detail)
    if not kept: return None,dict(reason='No retained candidate',rejected_seeds=rejected,diagnostics=diagnostics)
    if not flexible and len(kept)!=len(seeds): return None,dict(reason='Child lost during overlap resolution',diagnostics=diagnostics)
    return assigned,dict(reason='proposal',children=kept,rejected_seeds=rejected,diagnostics=diagnostics,
        residual_pixels=int((parent&(assigned==0)).sum()),added_foreground_pixels=int(((assigned>0)&~parent).sum()))


def infer(project,points_path,checkpoint,output,device='cuda:0',min_area=30,min_score=.8,predictor=None,prompt_mode='point-box'):
    project,output=Path(project).resolve(),Path(output).resolve();m=sealed(project/'project.json')
    if output.exists(): raise FileExistsError('Use a new output directory')
    if min_area<1 or not 0<=min_score<=1: raise ValueError('Invalid proposal thresholds')
    if prompt_mode not in ('point-box','point-only'): raise ValueError('Unknown prompt mode')
    hashes={str(p):digest(p) for p in (project/'project.json',Path(points_path).resolve(),Path(checkpoint).resolve())}
    selection=read(points_path)
    if selection['project_sha256']!=digest(project/'project.json'): raise ValueError('Points belong to another project')
    regions={r['id']:r for r in m['regions']};items=selection['instances'];seen=set()
    if not items: raise ValueError('No selected instances')
    # Validate all annotations before loading SAM.
    with rasterio.open(m['instances']) as src,rasterio.open(m['image']) as image:
        for item in items:
            lab=item['id']
            if lab in seen or lab not in regions: raise ValueError('Unknown or duplicate instance')
            seen.add(lab);x0,y0,x1,y1=regions[lab]['xyxy'];parent=src.read(1,window=Window(x0,y0,x1-x0,y1-y0))==lab
            pts=item['points'];win=Window(x0,y0,x1-x0,y1-y0)
            original=src.read(1,window=win);valid=np.all(image.read_masks(m['rgb_bands'],window=win)>0,axis=0)
            valid &= np.isfinite(image.read(m['rgb_bands'],window=win)).all(axis=0)
            if not 1<=len(pts)<=8 or len({tuple(p) for p in pts})!=len(pts): raise ValueError('Need 1 to 8 distinct points')
            item['point_roles']=[]
            for number,p in enumerate(pts,1):
                if len(p)!=2 or any(type(v) is not int for v in p): raise ValueError('Integer pixel points required')
                x,y=p
                if not (0<=y<parent.shape[0] and 0<=x<parent.shape[1] and valid[y,x]):
                    raise ValueError(f'Instance {lab}, point {number}: outside valid image context; coordinate ({x},{y})')
                actual=int(original[y,x]);kind='parent' if actual==lab else 'background' if actual==0 else 'other_instance'
                item['point_roles'].append(dict(point=number,source_id=actual,kind=kind))
                print(f'Instance {lab}, point {number}: {kind}, existing ID={actual}',flush=True)
    if m['max_source_id']+sum(len(r['points']) for r in items)>np.iinfo(np.uint32).max: raise ValueError('ID overflow')
    if predictor is None:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'segment-anything'))
        import torch
        from segment_anything import SamPredictor,sam_model_registry
        model=sam_model_registry['vit_b'](checkpoint=str(checkpoint));model.to(device=device).eval();predictor=SamPredictor(model)
    output.mkdir(parents=True);proposals=[];next_id=m['max_source_id']+1
    with rasterio.open(m['image']) as image,rasterio.open(m['instances']) as src:
        for item in items:
            lab=item['id'];row=regions[lab];x0,y0,x1,y1=row['xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
            original=src.read(1,window=win);parent=original==lab;rgb=rgb_read(image,win,m['rgb_bands'],m['rgb_range'])
            valid=np.all(image.read_masks(m['rgb_bands'],window=win)>0,axis=0)&np.isfinite(image.read(m['rgb_bands'],window=win)).all(axis=0)
            from .split_diagnostics import image_sink
            sink=image_sink(output/'diagnostics',lab,rgb)
            patch,detail=split_candidate(rgb,parent,item['points'],predictor,min_area,min_score,source=original,valid=valid,diagnostic_sink=sink,prompt_mode=prompt_mode)
            entry=dict(id=lab,xyxy=row['xyxy'],points=item['points'],point_roles=item['point_roles'],
                       detail=detail,status='rejected',editable='parent_and_background')
            shutil.copyfile(project/f'{lab}_target.png',output/f'{lab}_before.png')
            if patch is not None:
                ids=list(range(next_id,next_id+len(detail['children'])));next_id+=len(ids)
                updated=original.astype(np.uint32).copy()
                for child,global_id in zip(detail['children'],ids):
                    child['new_id']=global_id;updated[patch==child['seed']+1]=global_id
                path=output/f'{lab}_proposal.tif';save_labels(path,updated,dict(src.profile,width=x1-x0,height=y1-y0,transform=src.window_transform(win)))
                display=rgb.copy();protected=(original!=0)&~parent;display[protected]=(display[protected]*.3).astype(np.uint8)
                colors=[[255,80,80],[50,200,255],[100,255,100],[255,180,30],[220,100,255],[255,100,180],[120,200,130],[100,130,255]]
                for child,color in zip(detail['children'],colors):
                    mask=patch==child['seed']+1;display[mask]=(.5*display[mask]+.5*np.array(color)).astype(np.uint8)
                save_png(output/f'{lab}_after.png',display)
                entry.update(status='proposal',child_ids=ids,path=str(path),sha256=digest(path))
            proposals.append(entry)
    sealed(project/'project.json')
    if any(digest(p)!=value for p,value in hashes.items()): raise ValueError('Inputs changed during inference')
    report=dict(status='ready_for_review',operation='parent_refinement_and_background_completion',
        project=str(project),project_sha256=digest(project/'project.json'),
        proposals=proposals,min_area=min_area,min_score=min_score,prompt_mode=prompt_mode,input_sha256=hashes,ground_truth_used=False)
    (output/'proposals.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cards=[]
    for r in proposals:
        lab=r['id'];card=f'<h2>源实例 {lab}</h2><img src="{lab}_before.png" style="max-width:45%">'
        if r['status']=='proposal':
            card+=f'<img src="{lab}_after.png" style="max-width:45%"><p>彩色为候选新实例，原实例剩余像素保留。剩余：{r["detail"]["residual_pixels"]}，新增前景：{r["detail"].get("added_foreground_pixels",0)} 像素。</p><label><input type="checkbox" id="a{lab}">确认接受此候选（包括背景补充分割）</label>'
        else: card+='<p>本组没有保留候选，原实例保持不变。请查看逐点诊断，区分模型输出与筛选原因。</p>'
        import html
        for role in r['point_roles']:
            label={'parent':'当前实例内，参与细分','background':'原预测背景，参与补充候选','other_instance':'已有其他实例，保持保护；如需修改请打开对应ID'}[role['kind']]
            card+=f'<p>点{role["point"]}：{label}，原ID={role["source_id"]}</p>'
        for rejected in r['detail'].get('rejected_seeds',[]):
            card+=f'<p>点{rejected["seed"]+1}未生成候选：{html.escape(rejected["reason"])}</p>'
        card+='<p><a href="diagnostics.html">查看逐点SAM原始候选、分数及具体拒绝原因</a></p>'
        cards.append(card)
    config=json.dumps(dict(hash=digest(output/'proposals.json'),ids=[r['id'] for r in proposals if r['status']=='proposal']))
    script='const c='+config+''';function download(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({proposals_sha256:c.hash,accepted_ids:c.ids.filter(id=>document.getElementById('a'+id).checked)},null,2)],{type:'application/json'}));a.download='split_approvals.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}'''
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>拆分候选确认</h1><p>提示模式：'+prompt_mode+'。未勾选即保留原实例。不看GT后挑选。</p><button onclick="download()">导出确认JSON</button>'+''.join(cards)+'<script>'+script+'</script>',encoding='utf-8')
    from .split_diagnostics import write_report
    write_report(output,proposals,min_area,min_score)
    lines=[f'Prompt mode: {prompt_mode}; min_score={min_score}; min_area={min_area}; no GT read.',
           'Retained candidates are proposals, not confirmed correct parcels.']
    for p in proposals:
        lines.append(f"Instance {p['id']}: {p['status']}")
        for seed in p['detail'].get('diagnostics',[]):
            scores_text=','.join('NA' if c['score'] is None else f"{c['score']:.4f}" for c in seed['candidates'])
            lines.append(f"  Point {seed['seed']+1}: {seed['status']}; scores=[{scores_text}]; box={seed['box']}")
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    print(f'Open {output}/review.html; {sum(r["status"]=="proposal" for r in proposals)} proposals; no GT read')
    return report


def apply(proposals_dir,approvals_path,output):
    proposals_dir,output=Path(proposals_dir).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new output directory')
    r=sealed(proposals_dir/'proposals.json');project=Path(r['project']);m=sealed(project/'project.json')
    a=read(approvals_path)
    if a['proposals_sha256']!=digest(proposals_dir/'proposals.json'): raise ValueError('Approval belongs to another proposal set')
    accepted=a['accepted_ids'];available={p['id']:p for p in r['proposals'] if p['status']=='proposal'}
    if len(set(accepted))!=len(accepted) or not set(accepted)<=set(available): raise ValueError('Invalid approved IDs')
    for p in r['proposals']:
        if p['status']=='proposal' and digest(p['path'])!=p['sha256']: raise ValueError('Proposal changed')
    # Preflight all accepted edits, including background additions and cross-context conflicts.
    edits=[]
    with rasterio.open(m['instances']) as src:
        for lab in accepted:
            p=available[lab];x0,y0,x1,y1=p['xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
            original=src.read(1,window=win)
            with rasterio.open(p['path']) as patch:
                if patch.shape!=original.shape or not patch.transform.almost_equals(src.window_transform(win)):
                    raise ValueError('Proposal grid changed')
                candidate=patch.read(1)
            protected=(original!=lab)&(original!=0) if p.get('editable')=='parent_and_background' else original!=lab
            if np.any(candidate[protected]!=original[protected]): raise ValueError('Proposal overwrites another existing instance')
            change=candidate!=original
            if np.any(change&~np.isin(candidate,p['child_ids'])): raise ValueError('Unexpected deletion or unallocated new ID')
            for old,old_mask,_ in edits:
                a,b,c,d=old['xyxy'];left,top,right,bottom=max(a,x0),max(b,y0),min(c,x1),min(d,y1)
                if left<right and top<bottom and np.any(old_mask[top-b:bottom-b,left-a:right-a]&change[top-y0:bottom-y0,left-x0:right-x0]):
                    raise ValueError('Approved proposals overlap in their write pixels; approve one, then prepare a new round from its exported result')
            edits.append((p,change,candidate))
    output.mkdir(parents=True);mapping=[]
    with rasterio.open(m['instances']) as src:
        profile=src.profile.copy();profile.update(dtype='uint32',nodata=None,compress='deflate')
        with rasterio.open(output/'instance.tif','w',**profile) as dst:
            for _,win in src.block_windows(1): dst.write(src.read(1,window=win).astype(np.uint32),1,window=win)
            for p,mask,candidate in edits:
                lab=p['id'];x0,y0,x1,y1=p['xyxy']
                # Write only changed spans, preserving other accepted edits in overlapping contexts.
                for yy in range(mask.shape[0]):
                    cols=np.flatnonzero(mask[yy])
                    for chunk in np.split(cols,np.where(np.diff(cols)>1)[0]+1):
                        if len(chunk): dst.write(candidate[yy:yy+1,chunk[0]:chunk[-1]+1],1,window=Window(x0+int(chunk[0]),y0+yy,len(chunk),1))
                mapping.append(dict(parent_id=lab,child_ids=p['child_ids'],residual_pixels=p['detail']['residual_pixels'],
                    added_foreground_pixels=p['detail'].get('added_foreground_pixels',0)))
    sealed(project/'project.json')
    report=dict(status='complete',source_instances=m['instances'],source_sha256=digest(m['instances']),
        output_sha256=digest(output/'instance.tif'),accepted=mapping,ground_truth_used=False,
        proposals_sha256=digest(proposals_dir/'proposals.json'),approvals=read(approvals_path),
        scope='Full-image export; approved parent/background edits only; other existing IDs and unassigned pixels preserved')
    (output/'split_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Exported {output}/instance.tif; {len(mapping)} approved parent splits')
    return report


def evaluate(result_dir,gt_path,roi_path,output,allow=False,tolerance=2):
    from .evaluate_tiles import roi_counts
    from .metrics import scores
    from .tiles import mask_bounds
    result_dir,output=Path(result_dir).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new output directory')
    if tolerance<0: raise ValueError('Negative boundary tolerance')
    report=read(result_dir/'split_result.json')
    if report['status']!='complete': raise ValueError('Expected completed split export')
    for path,expected in ((report['source_instances'],report['source_sha256']),
                           (result_dir/'instance.tif',report['output_sha256'])):
        if digest(path)!=expected: raise ValueError('Committed split result or source changed')
    with rasterio.open(report['source_instances']) as src,rasterio.open(result_dir/'instance.tif') as edited,\
         rasterio.open(gt_path) as gt,rasterio.open(roi_path) as roi:
        for other in (edited,gt,roi): aligned(src,other,allow)
        if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
            raise ValueError('Expected integer GT IDs with NoData=None/0')
        x0,y0,x1,y1=mask_bounds(roi,tolerance+1);win=Window(x0,y0,x1-x0,y1-y0)
        raw=roi.read(1,window=win);mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
        truth=gt.read(1,window=win)
        if np.any(truth<0): raise ValueError('Negative GT IDs')
        entries={}
        for name,data in (('before',src.read(1,window=win)),('after',edited.read(1,window=win))):
            counts=roi_counts(data,truth,mask,tolerance);entries[name]=dict(scores(counts),counts=counts)
    output.mkdir(parents=True)
    result=dict(evaluation=entries,split_result_sha256=digest(result_dir/'split_result.json'),
                gt_sha256=digest(gt_path),roi_sha256=digest(roi_path),boundary_tolerance=tolerance,
                warning='ROI-clipped instance metrics; developer evaluation; independence and source provenance unverified')
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=[result['warning']]
    for name,e in entries.items():
        lines.append(f"{name}: IoU={e['mask_iou']:.6f}, boundary_F1={e['boundary_f1']:.6f}, instance_F1={e['instance_f1']:.6f}, PQ={e['pq']:.6f}")
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--image',type=Path);p.add_argument('--instances',type=Path)
    p.add_argument('--run-dir',type=Path,help='Read full source image/prediction and ROI paths from an existing window run, not its cropped output')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--instance-ids',type=int,nargs='+');p.add_argument('--eval-mask',type=Path)
    p.add_argument('--context',type=int,default=128);p.add_argument('--max-side',type=int,default=4096)
    p.add_argument('--rgb-bands',type=int,nargs=3);p.add_argument('--rgb-range',type=float,nargs=2);p.add_argument('--allow-crs-mismatch',action='store_true')
    p=sub.add_parser('infer');p.add_argument('--project-dir',type=Path,required=True);p.add_argument('--points',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--min-area',type=int,default=30);p.add_argument('--min-score',type=float,default=.8)
    p.add_argument('--prompt-mode',choices=['point-box','point-only'],default='point-box',help='point-only removes the merged-instance box; all other rules unchanged')
    p=sub.add_parser('apply');p.add_argument('--proposals-dir',type=Path,required=True);p.add_argument('--approvals',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--eval-mask',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-crs-mismatch',action='store_true');p.add_argument('--boundary-tolerance',type=int,default=2)
    a=parser.parse_args()
    if a.command=='prepare':
        if a.run_dir:
            source=read(a.run_dir/'manifest.json')
            if source.get('status')!='complete': raise ValueError('Expected completed source run')
            if a.image or a.instances: raise ValueError('Use --run-dir or explicit --image/--instances, not both')
            p=source['parameters'];a.image=Path(p['image']);a.instances=Path(p['prediction'])
            for key in ('prediction','eval_mask'):
                if digest(p[key])!=source['hashes'][key]: raise ValueError('Source run inputs changed')
            a.eval_mask=a.eval_mask or Path(p['eval_mask']);a.rgb_bands=a.rgb_bands or p.get('rgb_bands',[1,2,3])
            a.rgb_range=a.rgb_range or p.get('rgb_range');a.allow_crs_mismatch |= p.get('allow_crs_mismatch',False)
        if not a.image or not a.instances: raise ValueError('Provide --run-dir or both --image and --instances')
        prepare(a.image,a.instances,a.output_dir,a.instance_ids,a.eval_mask,a.context,a.max_side,a.rgb_bands or [1,2,3],a.rgb_range,a.allow_crs_mismatch)
    elif a.command=='infer':infer(a.project_dir,a.points,a.checkpoint,a.output_dir,a.device,a.min_area,a.min_score,prompt_mode=a.prompt_mode)
    elif a.command=='apply':apply(a.proposals_dir,a.approvals,a.output_dir)
    else:evaluate(a.result_dir,a.gt,a.eval_mask,a.output_dir,a.allow_crs_mismatch,a.boundary_tolerance)


if __name__=='__main__':main()
