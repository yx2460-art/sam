"""Fixed-region fusion evaluation, followed by optional human SAM refinement and re-evaluation."""
import argparse
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import Window

from .evaluate_tiles import roi_counts
from .manual_fusion import read
from .metrics import scores
from .raster import save_labels
from .refine_conflicts import save_mask
from .refuse_tiles import refusion
from .report_regions import overlay,save_png
from .run_tiles import aligned,digest
from .split_instances import sealed,rgb_read,infer as split_infer,apply as split_apply


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def load(directory):
    directory=Path(directory).resolve();m=sealed(directory/'experiment.json')
    if m['status']!='ready': raise ValueError('Expected completed regional preparation')
    return directory,m


def prepare(run_dir,output,instance_id=None,bounds=None,padding=0,context=128):
    run_dir,output=Path(run_dir).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new output directory')
    if (instance_id is None)==(bounds is None): raise ValueError('Specify exactly one --instance-id or --bounds x y width height')
    if instance_id is not None and instance_id<=0: raise ValueError('Source instance ID must be positive; use --bounds for a background region')
    if padding<0 or context<0: raise ValueError('Negative region padding/context')
    source=read(run_dir/'manifest.json');params=source['parameters']
    if source.get('status')!='complete': raise ValueError('Expected completed window run')
    allow=params.get('allow_crs_mismatch',False);global_max=0;box=None
    with rasterio.open(params['prediction']) as base:
        for _,win in base.block_windows(1):
            a=base.read(1,window=win);global_max=max(global_max,int(a.max()))
            if instance_id is not None:
                yy,xx=np.where(a==instance_id)
                if xx.size:
                    b=[int(win.col_off+xx.min()),int(win.row_off+yy.min()),int(win.col_off+xx.max()+1),int(win.row_off+yy.max()+1)]
                    box=b if box is None else [min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3])]
        if bounds is not None:
            x,y,w,h=bounds
            if w<=0 or h<=0: raise ValueError('Bounds width/height must be positive')
            box=[x,y,x+w,y+h]
        if box is None: raise ValueError('Requested source instance absent')
        box=[box[0]-padding,box[1]-padding,box[2]+padding,box[3]+padding]
        if not (0<=box[0]<box[2]<=base.width and 0<=box[1]<box[3]<=base.height):
            raise ValueError('Selected region outside image; choose explicit in-image bounds')
    # Full cached-fusion validation, without reading GT.
    refusion(run_dir,output/'fusion_cache',methods=['center','union','intersection'])
    cx0,cy0,cx1,cy1=source['crop_xyxy']
    if not (cx0<=box[0]<box[2]<=cx1 and cy0<=box[1]<box[3]<=cy1):
        raise ValueError('Region outside cached prediction coverage; run windows covering this region first')
    # Context is for display/inference only. Scoring and final writes stay in the fixed region.
    crop=[max(cx0,box[0]-context),max(cy0,box[1]-context),min(cx1,box[2]+context),min(cy1,box[3]+context)]
    x0,y0,x1,y1=crop;win=Window(x0,y0,x1-x0,y1-y0)
    with rasterio.open(params['image']) as image,rasterio.open(params['prediction']) as base,\
         rasterio.open(params['eval_mask']) as roi,rasterio.open(params['coverage']) as coverage:
        for other in (base,roi,coverage): aligned(image,other,allow)
        profile=dict(base.profile,width=x1-x0,height=y1-y0,transform=image.window_transform(win))
        rgb=rgb_read(image,win,params.get('rgb_bands',[1,2,3]),params.get('rgb_range'))
        valid=np.all(image.read_masks(params.get('rgb_bands',[1,2,3]),window=win)>0,axis=0)&(coverage.read(1,window=win)>0)
        raw=roi.read(1,window=win);mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
        rectangle=np.zeros(mask.shape,dtype=bool)
        rectangle[box[1]-y0:box[3]-y0,box[0]-x0:box[2]-x0]=True
        mask &= rectangle
        if not mask.any(): raise ValueError('Selected region has no pixels in the fixed annotation/evaluation ROI')
        if np.any(mask&~valid): raise ValueError('Scoring region includes invalid image/coverage pixels')
        image_path=output/'region_image.tif'
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(image_path,'w',**dict(profile,count=3,dtype='uint8',nodata=None)) as dst:
                dst.write(np.moveaxis(rgb,-1,0));dst.write_mask(valid.astype(np.uint8)*255)
        save_mask(output/'region_mask.tif',mask,profile)
        candidates={'original':output/'original.tif'};arrays={'original':base.read(1,window=win)}
        save_labels(candidates['original'],arrays['original'],profile)
        cache_win=Window(x0-cx0,y0-cy0,x1-x0,y1-y0)
        for method in ('center','union','intersection'):
            with rasterio.open(output/'fusion_cache'/method/'instance.tif') as src: data=src.read(1,window=cache_win)
            arrays[method]=data;candidates[method]=output/f'{method}.tif';save_labels(candidates[method],data,profile)
        save_png(output/'rgb.png',rgb)
        for name,data in arrays.items(): save_png(output/f'{name}.png',overlay(rgb,data,mask))
    inputs=[run_dir/'manifest.json',Path(params['image']),Path(params['prediction']),Path(params['eval_mask']),Path(params['coverage']),
            image_path,output/'region_mask.tif',*candidates.values()]
    m=dict(status='ready',source_run=str(run_dir),requested_instance_id=instance_id,region_xyxy=box,context_xyxy=crop,
        scoring_pixels=int(mask.sum()),rectangle_pixels=int(rectangle.sum()),global_max_id=global_max,
        checkpoint_sha256=source['hashes']['checkpoint'],
        image=str(image_path),mask=str(output/'region_mask.tif'),full_reference=params['prediction'],candidates={k:str(p) for k,p in candidates.items()},
        allow_crs_mismatch=allow,boundary_tolerance=params.get('boundary_tolerance',2),
        input_sha256={str(p):digest(p) for p in inputs},
        scope='Fixed rectangle intersected with original evaluation ROI; context excluded from scoring/writes; developer experiment')
    write_json(output/'experiment.json',m)
    panels=''.join(f'<figure><img src="{name}.png"><figcaption>{label}</figcaption></figure>' for name,label in
        [('rgb','原始影像'),('original','原始实例'),('center','中心优先参考'),('intersection','求交'),('union','求并')])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>固定区域融合对照</h1>'
        f'<p>选定原图矩形 {box}，评分像素 {m["scoring_pixels"]}。变暗区域不参与本轮评分及最终修改。无GT叠加。</p>'
        '<style>figure{display:inline-block;width:44%;vertical-align:top}img{width:100%}</style>'+panels,encoding='utf-8')
    print(f'Region ready: {output}/index.html; scoring pixels={m["scoring_pixels"]}; no GT read')
    return m


def fusion_scores(experiment,gt_path,output,result_dir=None):
    experiment,m=load(experiment);output=Path(output).resolve();gt_path=Path(gt_path).resolve()
    if output.exists(): raise FileExistsError('Use a new score directory')
    paths=dict(m['candidates']);interaction=None;parent_score=None
    if result_dir:
        result_dir=Path(result_dir).resolve();interaction=sealed(result_dir/'region_result.json')
        if interaction['experiment_sha256']!=digest(experiment/'experiment.json'): raise ValueError('Result belongs to another region')
        start=sealed(Path(interaction['interaction_project'])/'workflow.json')
        parent_score=sealed(Path(start['fusion_scores'])/'scores.json')
        if parent_score['gt_sha256']!=digest(gt_path): raise ValueError('GT differs from the first fusion evaluation')
        if start.get('previous_result'):
            previous=Path(start['previous_result']);sealed(previous/'region_result.json')
            paths['previous_interactive']=str(previous/'region_result.tif')
        paths['interactive']=str(result_dir/'region_result.tif')
    x0,y0,x1,y1=m['context_xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
    with rasterio.open(m['image']) as image,rasterio.open(gt_path) as gt,rasterio.open(m['mask']) as roi,\
         rasterio.open(m['full_reference']) as reference:
        aligned(reference,gt,m['allow_crs_mismatch'])
        expected=SimpleNamespace(width=win.width,height=win.height,crs=gt.crs,transform=gt.window_transform(win),count=1,name='GT region window')
        # CRS override follows the pre-existing source policy; dimensions/transform must still agree.
        aligned(image,expected,m['allow_crs_mismatch'])
        if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0): raise ValueError('Expected integer GT IDs, NoData=None/0')
        truth=gt.read(1,window=win)
        if np.any(truth<0): raise ValueError('Negative GT IDs')
        mask=roi.read(1)>0;entries={}
        for name,p in paths.items():
            with rasterio.open(p) as src: aligned(image,src,False);data=src.read(1)
            counts=roi_counts(data,truth,mask,m['boundary_tolerance']);entries[name]=dict(scores(counts),counts=counts)
    if parent_score:
        for name in m['candidates']:
            if entries[name]!=parent_score['evaluation'][name]: raise ValueError('Original fusion scores changed')
    output.mkdir(parents=True)
    score=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),gt_sha256=digest(gt_path),
        evaluation=entries,region_xyxy=m['region_xyxy'],scoring_pixels=m['scoring_pixels'],
        interactive_start=None if not interaction else interaction['start_method'],
        interaction_round=None if not interaction else interaction.get('interaction_round',1),
        warning='Same fixed regional ROI; GT used for development evaluation; independence unverified',
        input_sha256={str(experiment/'experiment.json'):digest(experiment/'experiment.json'),**{str(p):digest(p) for p in paths.values()}})
    write_json(output/'scores.json',score)
    rows=[dict(method=name,**{k:v for k,v in entry.items() if k!='counts'}) for name,entry in entries.items()]
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[score['warning'],f"Scoring pixels={m['scoring_pixels']}; region={m['region_xyxy']}"]
    for row in rows: lines.append(f"{row['method']}: IoU={row['mask_iou']:.6f}, boundary_F1={row['boundary_f1']:.6f}, instance_F1={row['instance_f1']:.6f}, PQ={row['pq']:.6f}")
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
    return score


def start(experiment,score_dir,method,reason,output,target_id=None,interaction_mode='separate-parcels',from_result=None):
    if interaction_mode not in ('separate-parcels','single-target'): raise ValueError('Unknown interaction mode')
    experiment,m=load(experiment);score_dir=Path(score_dir).resolve();output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new interaction directory')
    first=sealed(score_dir/'scores.json')
    if first['experiment_sha256']!=digest(experiment/'experiment.json') or 'interactive' in first['evaluation']:
        raise ValueError('Use the first fusion evaluation for this fixed region')
    if method not in ('union','intersection','center','original') or not reason.strip(): raise ValueError('Specify starting fusion and reason')
    source_path=Path(m['candidates'][method]);previous=None;round_number=1;extra_inputs=[]
    if from_result is not None:
        previous=Path(from_result).resolve();prior=sealed(previous/'region_result.json')
        if prior['experiment_sha256']!=digest(experiment/'experiment.json'): raise ValueError('Previous result belongs to another region')
        if prior['start_method']!=method: raise ValueError('Keep the original fusion method when continuing a result')
        prior_workflow=sealed(Path(prior['interaction_project'])/'workflow.json')
        if Path(prior_workflow['fusion_scores']).resolve()!=score_dir: raise ValueError('Keep the same first fusion evaluation')
        source_path=previous/'region_result.tif';round_number=prior.get('interaction_round',1)+1
        extra_inputs=[previous/'region_result.json']
    with rasterio.open(m['image']) as image,rasterio.open(source_path) as src,rasterio.open(m['mask']) as roi:
        aligned(image,src,False)
        data=src.read(1);mask=roi.read(1)>0;rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
        ids,counts=np.unique(data[mask],return_counts=True);positive=ids>0
        anchor=int(ids[positive][np.argmax(counts[positive])]) if positive.any() else 0
        if m.get('requested_instance_id') in ids[positive]: anchor=m['requested_instance_id']
        if target_id is not None:
            if target_id not in ids: raise ValueError('Target ID absent from the fixed scoring region')
            anchor=int(target_id)
        ownership=[]
        for line in data:
            starts=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1];ownership.append([[int(x),int(line[x])] for x in starts])
        output.mkdir(parents=True);save_png(output/f'{anchor}_target.png',overlay(rgb,data,mask))
        project=dict(status='ready',image=m['image'],instances=str(source_path),
            regions=[dict(id=anchor,xyxy=[0,0,image.width,image.height])],max_source_id=max(m['global_max_id'],int(data.max())),
            global_xy_offset=m['context_xyxy'][:2],
            rgb_bands=[1,2,3],rgb_range=None,allow_crs_mismatch=False,
            input_sha256={str(p):digest(p) for p in (m['image'],source_path,m['mask'],*extra_inputs)},
            protocol='Fixed-region post-fusion positive interaction; protected existing IDs; point-only')
        write_json(output/'project.json',project)
        # This page annotates the fixed region; the scoring mask is checked again before inference.
        allowed=[]
        for line in mask:
            changes=np.diff(np.r_[0,line.astype(np.int8),0]);starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
            allowed.append([[int(a),int(b)] for a,b in zip(starts,ends)])
        config=json.dumps(dict(hash=digest(output/'project.json'),id=anchor,ownership=ownership,allowed=allowed,mode=interaction_mode))
        page=f'<!doctype html><meta charset="utf-8"><h1>SAM 交互后处理：田块实例提取</h1><p>已有结果：{method}；本次目标ID：{anchor}。原因：{html.escape(reason)}。</p>'
        page+='<p>本页只记录正点，不重新求交或求并，也不会直接运行SAM。先在想单独提取的一个田块内部点一次；导出 region_points.json 后运行 infer，才能查看SAM候选。</p>'
        page+='<p>也可在多个不同田块内各点一次，共1至8点。只在未变暗的固定区域内标点；其他已有ID受保护，若需修改它们，请用 start --target-id 对应ID 新建项目。可撤销，不看GT边界选点。</p>'
        if interaction_mode=='single-target':
            page=f'<!doctype html><meta charset="utf-8"><h1>SAM 单田块正负点交互</h1><p>起点：{method}；目标ID：{anchor}；原因：{html.escape(reason)}。</p>'
            page+='<p>本页只提取一个田块。正点（红色）放在目标内部，负点（蓝色）放在被误包含的邻近田块。所有点共同输入SAM，最多8点，至少一个正点。不重新求并，不看GT。仅亮区可标点，其他已有ID允许负点但不会被覆盖。</p>'
            page+='<label>点击类型：<select id="polarity"><option value="1">正点：保留目标</option><option value="0">负点：排除邻块</option></select></label><p>补点重试：恢复本项目上一次导出的JSON，补点后再次导出；infer使用新的输出目录。</p><input type="file" accept=".json" onchange="restore(this)">'
        page+=f'<canvas id="canvas" width="{image.width}" height="{image.height}" style="max-width:96vw;height:auto"></canvas><p id="status"></p><button onclick="undo()">撤销</button><button onclick="download()">导出 region_points.json</button>'
        if previous:
            page+=f'<p>累计交互第{round_number}轮：从已确认结果 {html.escape(str(previous))} 继续。本轮仍修改目标ID {anchor} 的剩余部分，先前提取的新实例受保护。请选下一条尚未提取的田块。</p>'
    script=r'''
const c=CONFIG,points=[],labels=[],canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),img=new Image(),began=Date.now();
function draw(){ctx.drawImage(img,0,0);points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.fillStyle=labels[i]===0?'#0088ff':'red';ctx.fill();ctx.font='22px sans-serif';ctx.fillStyle='white';ctx.fillText((labels[i]===0?'-':'+')+(i+1),p[0]+8,p[1]);});
document.getElementById('status').textContent=points.map((p,i)=>{let id=0;for(const r of c.ownership[p[1]]||[]){if(r[0]>p[0])break;id=r[1];}return '点'+(i+1)+': '+(id===0?'融合结果背景':id===c.id?'当前目标ID '+id:'其他已有ID '+id+'（保护）');}).join('；');}
img.onload=draw;img.src=c.id+'_target.png';canvas.onclick=e=>{if(points.length>=8){alert('最多8点');return;}const r=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=Math.floor((e.clientY-r.top)*canvas.height/r.height);
if(!(c.allowed[y]||[]).some(([a,b])=>x>=a&&x<b)){alert('这里是本轮固定区域以外的上下文，不会回写。请在未变暗区域点选。');return;}
let owner=0;for(const r of c.ownership[y]||[]){if(r[0]>x)break;owner=r[1];}
const label=c.mode==='single-target'?Number(document.getElementById('polarity').value):1;
if(label===1&&owner!==0&&owner!==c.id){alert('此处属于已有实例ID '+owner+'，本项目保护该实例。若要修改它，请用 start --target-id '+owner+' 新建交互项目；本次点击未记录。');return;}
if(points.some(p=>p[0]===x&&p[1]===y)){alert('此坐标已标点，请撤销后重选');return;}
points.push([x,y]);labels.push(label);draw();};
function undo(){points.pop();labels.pop();draw();}
async function restore(input){try{const v=JSON.parse(await input.files[0].text()),r=v.instances[0];
if(v.project_sha256!==c.hash||v.interaction_mode!=='single-target'||v.instances.length!==1||r.id!==c.id||!Array.isArray(r.points)||!Array.isArray(r.point_labels)||r.points.length!==r.point_labels.length||r.points.length>8)throw Error('请选择本项目导出的正负点JSON');
if(r.points.some((p,i)=>!Array.isArray(p)||p.length!==2||!p.every(Number.isInteger)||![0,1].includes(r.point_labels[i])||!(c.allowed[p[1]]||[]).some(([a,b])=>p[0]>=a&&p[0]<b)))throw Error('点格式或区域不匹配');
points.splice(0,points.length,...r.points);labels.splice(0,labels.length,...r.point_labels);draw();}catch(e){alert(e.message);}}
function download(){if(!points.length||!labels.includes(1)){alert('至少需要一个正点');return;}const row={id:c.id,points},v={project_sha256:c.hash,instances:[row],browser_wall_seconds:(Date.now()-began)/1000};
if(c.mode==='single-target'){v.interaction_mode=c.mode;row.point_labels=labels;}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,2)],{type:'application/json'}));a.download='region_points.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
'''.replace('CONFIG',config)
    (output/'index.html').write_text(page+'<script>'+script+'</script>',encoding='utf-8')
    workflow=dict(status='ready',experiment=str(experiment),experiment_sha256=digest(experiment/'experiment.json'),
        fusion_scores=str(score_dir),start_method=method,reason=reason,target_id=anchor,interaction_mode=interaction_mode,mask=m['mask'],
        previous_result=None if previous is None else str(previous),interaction_round=round_number,
        input_sha256={str(p):digest(p) for p in (experiment/'experiment.json',score_dir/'scores.json',output/'project.json',*extra_inputs)})
    write_json(output/'workflow.json',workflow);print(f'Open {output}/index.html; start={method}; no GT read')
    return workflow


def infer(project,points,checkpoint,output,device='cuda:0',predictor=None):
    project=Path(project).resolve();w=sealed(project/'workflow.json');annotation=read(points)
    if annotation.get('interaction_mode','separate-parcels')!=w.get('interaction_mode','separate-parcels'): raise ValueError('Annotation interaction mode differs from project')
    _,m=load(w['experiment'])
    if digest(checkpoint)!=m['checkpoint_sha256']: raise ValueError('Use the same SAM checkpoint as the fusion source run')
    with rasterio.open(w['mask']) as src: mask=src.read(1)>0
    for item in annotation['instances']:
        for p in item['points']:
            if len(p)!=2 or any(type(v) is not int for v in p): raise ValueError('Integer point coordinates required')
            x,y=p
            if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]): raise ValueError('Point outside the fixed scoring/write region (dark area)')
    return split_infer(project,points,checkpoint,output,device=device,predictor=predictor,prompt_mode='point-only')


def apply(proposals,approvals,output):
    proposals,output=Path(proposals).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new result directory')
    p=sealed(proposals/'proposals.json');project=Path(p['project']);w=sealed(project/'workflow.json')
    experiment,m=load(w['experiment'])
    if not read(approvals).get('accepted_ids'): raise ValueError('No candidate accepted: tick the review checkbox and export approvals again')
    project_manifest=sealed(project/'project.json')
    split_apply(proposals,approvals,output/'unclipped_candidate')
    with rasterio.open(project_manifest['instances']) as src,rasterio.open(m['mask']) as roi,\
         rasterio.open(output/'unclipped_candidate'/'instance.tif') as candidate:
        before=src.read(1);data=before.copy();mask=roi.read(1)>0;edited=candidate.read(1);data[mask]=edited[mask]
        save_labels(output/'region_result.tif',data,src.profile)
        save_mask(output/'changed_pixels.tif',data!=before,src.profile)
    report=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),interaction_project=str(project),
        start_method=w['start_method'],reason=w['reason'],ground_truth_used_for_edits=False,
        changed_pixels=int((data!=before).sum()),scoring_pixels=int(mask.sum()),
        previous_result=w.get('previous_result'),interaction_round=w.get('interaction_round',1),
        prompt_points=sum(len(entry['points']) for entry in p['proposals']),
        scope='Region-context crop export; only fixed region mask modified; not a whole-image export',
        input_sha256={str(path):digest(path) for path in (project/'workflow.json',output/'region_result.tif',
            output/'unclipped_candidate'/'split_result.json')})
    write_json(output/'region_result.json',report);print(f'Exported {output}/region_result.tif; fixed write region only')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--instance-id',type=int);p.add_argument('--bounds',type=int,nargs=4);p.add_argument('--padding',type=int,default=0);p.add_argument('--context',type=int,default=128)
    p=sub.add_parser('score');p.add_argument('--experiment-dir',type=Path,required=True);p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--result-dir',type=Path)
    p=sub.add_parser('start');p.add_argument('--experiment-dir',type=Path,required=True);p.add_argument('--fusion-scores',type=Path,required=True)
    p.add_argument('--target-id',type=int,help='Explicit existing ID to refine; other existing IDs remain protected')
    p.add_argument('--from-result',type=Path,help='Continue an approved regional result; preserve prior edits and allocate fresh IDs')
    p.add_argument('--interaction-mode',choices=['separate-parcels','single-target'],default='separate-parcels',help='single-target jointly prompts one parcel using positive and negative points')
    p.add_argument('--fusion',choices=['union','intersection','center','original'],required=True);p.add_argument('--reason',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('infer');p.add_argument('--project-dir',type=Path,required=True);p.add_argument('--points',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p=sub.add_parser('apply');p.add_argument('--proposals-dir',type=Path,required=True);p.add_argument('--approvals',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare': prepare(a.run_dir,a.output_dir,a.instance_id,a.bounds,a.padding,a.context)
    elif a.command=='score': fusion_scores(a.experiment_dir,a.gt,a.output_dir,a.result_dir)
    elif a.command=='start': start(a.experiment_dir,a.fusion_scores,a.fusion,a.reason,a.output_dir,a.target_id,a.interaction_mode,a.from_result)
    elif a.command=='infer': infer(a.project_dir,a.points,a.checkpoint,a.output_dir,a.device)
    else: apply(a.proposals_dir,a.approvals,a.output_dir)


if __name__=='__main__': main()
