"""Offline human selection of cached union/intersection candidates, followed by separate scoring."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .evaluate_tiles import roi_counts
from .metrics import scores
from .raster import save_labels
from .refuse_tiles import refusion
from .report_regions import overlay,save_png
from .run_tiles import aligned,digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(project):
    project=Path(project).resolve()
    manifest=read(project/'project.json')
    if manifest.get('status')!='ready':
        raise ValueError('Expected a ready selection project')
    for path,value in manifest['input_sha256'].items():
        if digest(path)!=value:
            raise ValueError(f'Changed project input: {path}')
    return project,manifest


def prepare(run_dir,output_dir,size=256):
    run_dir,output_dir=Path(run_dir).resolve(),Path(output_dir).resolve()
    if size<=0: raise ValueError('Region size must be positive')
    if output_dir.exists(): raise FileExistsError(f'Use a new directory: {output_dir}')
    if output_dir==run_dir or run_dir in output_dir.parents:
        raise ValueError('Project must be outside source run')
    # Reuse the existing sealed-cache validator; never supply GT to this stage.
    refusion(run_dir,output_dir/'candidates',methods=['center','union','intersection'])
    source=read(run_dir/'manifest.json')
    params=source['parameters']
    x0,y0,x1,y1=source['crop_xyxy']
    window=Window(x0,y0,x1-x0,y1-y0)
    allow=params.get('allow_crs_mismatch',False)
    paths={m:output_dir/'candidates'/m/'instance.tif' for m in ('center','union','intersection')}
    with rasterio.open(params['image']) as image,rasterio.open(params['eval_mask']) as roi:
        aligned(image,roi,allow)
        raw=roi.read(1,window=window)
        domain=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=window)>0)
        labels={}
        for name,path in paths.items():
            with rasterio.open(path) as src: labels[name]=src.read(1)
        rows=[]; features=[]
        for y in range(0,domain.shape[0],size):
            for x in range(0,domain.shape[1],size):
                h,w=min(size,domain.shape[0]-y),min(size,domain.shape[1]-x)
                sl=np.s_[y:y+h,x:x+w]
                if not domain[sl].any(): continue
                index=len(rows)
                row=dict(id=index,x=x,y=y,x_full=x0+x,y_full=y0+y,width=w,height=h,
                         roi_pixels=int(domain[sl].sum()),
                         disagreement_pixels=int(((labels['union'][sl]!=labels['intersection'][sl])&domain[sl]).sum()))
                rows.append(row)
                rgb=image.read(params.get('rgb_bands',[1,2,3]),window=Window(x0+x,y0+y,w,h))
                if params.get('rgb_range'):
                    lo,hi=params['rgb_range']; rgb=np.clip((rgb.astype(float)-lo)*255/(hi-lo),0,255)
                rgb=np.moveaxis(rgb,0,-1).astype(np.uint8)
                save_png(output_dir/f'region_{index}_rgb.png',rgb)
                for method,data in labels.items():
                    save_png(output_dir/f'region_{index}_{method}.png',overlay(rgb,data[sl],domain[sl]))
                corners=[image.transform*(xx,yy) for xx,yy in
                         ((x0+x,y0+y),(x0+x+w,y0+y),(x0+x+w,y0+y+h),(x0+x,y0+y+h),(x0+x,y0+y))]
                features.append(dict(type='Feature',properties=row,
                                     geometry=dict(type='Polygon',coordinates=[corners])))
        if not rows: raise ValueError('Empty selection ROI')
        scale=min(1,1000/max(domain.shape));oh,ow=max(1,round(domain.shape[0]*scale)),max(1,round(domain.shape[1]*scale))
        overview=image.read(params.get('rgb_bands',[1,2,3]),window=window,out_shape=(3,oh,ow))
        if params.get('rgb_range'):
            lo,hi=params['rgb_range'];overview=np.clip((overview.astype(float)-lo)*255/(hi-lo),0,255)
        save_png(output_dir/'overview.png',np.moveaxis(overview,0,-1).astype(np.uint8))
        geo=dict(type='FeatureCollection',features=features)
        if image.crs:
            geo['crs']=dict(type='name',properties=dict(name=image.crs.to_string()))
        (output_dir/'regions.geojson').write_text(json.dumps(geo),encoding='utf-8')
        from .refine_conflicts import save_mask
        with rasterio.open(paths['center']) as src: profile=src.profile.copy()
        save_mask(output_dir/'write_domain.tif',domain,profile)
    inputs=[run_dir/'manifest.json',Path(params['image']),Path(params['prediction']),
            Path(params['eval_mask']),output_dir/'write_domain.tif',*paths.values()]
    manifest=dict(status='ready',source_run=str(run_dir),parameters=params,crop_xyxy=source['crop_xyxy'],
                  baseline='center',region_size=size,regions=rows,
                  candidates={m:str(p) for m,p in paths.items()},
                  input_sha256={str(p):digest(p) for p in inputs},
                  ground_truth_used=False,scope='Development ROI; fixed BsiNet IDs; human-assisted cached fusion')
    (output_dir/'project.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    project_hash=digest(output_dir/'project.json')
    cards=[]
    for row in rows:
        i=row['id']
        panels=''.join(f'<figure><img src="region_{i}_{m}.png"><figcaption>{label}</figcaption></figure>'
                       for m,label in [('rgb','原始影像'),('center','当前结果：中心优先'),
                                       ('intersection','求交候选'),('union','求并候选')])
        cards.append(f'<section id="r{i}"><h2>区域 {i} · 原图像素 ({row["x_full"]}, {row["y_full"]})'
                     f' · 分歧 {row["disagreement_pixels"]} 像素</h2><div class="panels">{panels}</div>'
                     f'<label>选择 <select id="choice{i}" onchange="record({i})">'
                     '<option value="keep">保留当前结果</option><option value="intersection">求交</option>'
                     '<option value="union">求并</option></select></label> '
                     f'<button onclick="review({i})">确认已查看此区域</button> '
                     f'<span id="state{i}">尚未查看，默认保留</span>'
                     f'<figure><img id="preview{i}" src="region_{i}_center.png">'
                     '<figcaption>当前选择预览（导出决策后需运行 apply 生成GeoTIFF）</figcaption></figure></section>')
    overview_svg=f'<svg viewBox="0 0 {ow} {oh}" style="width:min(100%,1000px)"><image href="overview.png" width="{ow}" height="{oh}"/>'
    for r in rows:
        overview_svg+=(f'<a href="#r{r["id"]}"><rect x="{r["x"]*ow/domain.shape[1]}" '
                       f'y="{r["y"]*oh/domain.shape[0]}" width="{r["width"]*ow/domain.shape[1]}" '
                       f'height="{r["height"]*oh/domain.shape[0]}" fill="transparent" stroke="yellow">'
                       f'<title>区域 {r["id"]} · 分歧 {r["disagreement_pixels"]} 像素</title></rect></a>')
    overview_svg+='</svg>'
    data=json.dumps(dict(project_sha256=project_hash,regions=[r['id'] for r in rows]))
    script=r'''
const config=CONFIG, key='manual-fusion-'+config.project_sha256;
let decisions=Object.fromEntries(config.regions.map(id=>[id,{id,choice:'keep',reviewed:false,changes:0}]));
let sessions=[], events=[], started=Date.now();
function validate(rows){if(!Array.isArray(rows)||rows.length!==config.regions.length)throw Error('区域数量不匹配');
const seen=new Set();for(const d of rows){if(!config.regions.includes(d.id)||seen.has(d.id)||
!['keep','union','intersection'].includes(d.choice)||typeof d.reviewed!=='boolean'||(!d.reviewed&&d.choice!=='keep'))throw Error('决策无效或区域重复');seen.add(d.id);}}
try {const saved=JSON.parse(localStorage.getItem(key));if(saved){validate(Object.values(saved.decisions));decisions=saved.decisions;sessions=saved.sessions||[];events=saved.events||[];}}catch(e){}
function store(){try{localStorage.setItem(key,JSON.stringify({decisions,sessions,events}));}catch(e){document.getElementById('message').textContent='浏览器无法保存本地进度，请导出JSON备份。';}}
function show(id){const d=decisions[id];document.getElementById('choice'+id).value=d.choice;
document.getElementById('state'+id).textContent=d.reviewed?'已确认':'尚未查看，默认保留';
document.getElementById('preview'+id).src='region_'+id+'_'+(d.choice==='keep'?'center':d.choice)+'.png';
const all=Object.values(decisions);document.getElementById('progress').textContent='已查看 '+all.filter(r=>r.reviewed).length+'/'+all.length+'，选择修改 '+all.filter(r=>r.choice!=='keep').length+' 个区域';}
function record(id){const d=decisions[id];d.choice=document.getElementById('choice'+id).value;
d.reviewed=true;d.changes++;d.last_action_utc=new Date().toISOString();events.push({id,action:'choose',choice:d.choice,utc:d.last_action_utc});show(id);store();}
function review(id){decisions[id].reviewed=true;decisions[id].last_action_utc=new Date().toISOString();events.push({id,action:'review',utc:decisions[id].last_action_utc});show(id);store();}
function jump(){const id=Number(document.getElementById('jump').value);if(!config.regions.includes(id)){document.getElementById('message').textContent='不存在该区域ID';return;}location.hash='r'+id;}
async function restore(file){if(!file)return;try{const data=JSON.parse(await file.text());
if(data.project_sha256!==config.project_sha256)throw Error('JSON来自其他选择项目');validate(data.decisions);
if(!Array.isArray(data.session_wall_seconds)||data.session_wall_seconds.some(v=>!Number.isFinite(v)||v<0)||!Array.isArray(data.events))throw Error('操作记录格式无效');
decisions=Object.fromEntries(data.decisions.map(d=>[d.id,d]));sessions=data.session_wall_seconds;events=data.events;started=Date.now();
config.regions.forEach(show);store();document.getElementById('message').textContent='已从JSON恢复进度';
}catch(e){document.getElementById('message').textContent='导入失败：'+e.message;}}
function download(){const elapsed=(Date.now()-started)/1000;sessions.push(elapsed);started=Date.now();store();
const payload={project_sha256:config.project_sha256,decisions:config.regions.map(id=>decisions[id]),
session_wall_seconds:sessions,events,exported_utc:new Date().toISOString(),timing_note:'Browser wall time includes pauses; self-reported, not verified active work'};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
a.download='decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
config.regions.forEach(show);
'''.replace('CONFIG',data)
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>人工选择融合实验</title>
<style>body{font-family:sans-serif;margin:20px}header{position:sticky;top:0;background:white;padding:12px;border:1px solid #888}
.panels{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:256px;image-rendering:pixelated}section{border-bottom:1px solid #aaa;padding:12px;scroll-margin-top:140px}button,select,input{font-size:16px}</style>
<header><b>人工选择求交／求并</b><p>仅依据原始影像判断，无GT。黄色为实例边界，评分区域外变暗。
未查看区域保留当前中心优先结果。每个区域独立选择，可反复修改；完成后导出决策。</p>
区域ID <input id="jump" type="number" min="0" value="0"><button onclick="jump()">定位区域</button>
<button onclick="download()">导出 decisions.json</button></header><p>点击总览中的黄色网格定位区域：</p>'''+overview_svg+''.join(cards)+'<script>'+script+'</script></html>'
    document=document.replace('</header>',
        '<label>恢复决策JSON <input type="file" accept=".json" onchange="restore(this.files[0])"></label>'
        '<p id="progress"></p><p id="message" role="status"></p></header>')
    (output_dir/'index.html').write_text(document,encoding='utf-8')
    print(f'Ready: {output_dir}/index.html; {len(rows)} regions. No GT read, no SAM inference.')
    return manifest


def apply(project,decisions_path,output_dir):
    project,m=verify(project)
    output_dir=Path(output_dir).resolve()
    if output_dir.exists(): raise FileExistsError(f'Use a new directory: {output_dir}')
    if project==output_dir or project in output_dir.parents:
        raise ValueError('Output must be outside the selection project')
    choices=read(decisions_path)
    if choices.get('project_sha256')!=digest(project/'project.json'):
        raise ValueError('Decisions belong to another project')
    records=choices.get('decisions',[])
    timing=choices.get('session_wall_seconds',[])
    events=choices.get('events',[])
    if (not isinstance(timing,list) or any(type(t) not in (int,float) or not np.isfinite(t) or t<0 for t in timing)
            or not isinstance(events,list)):
        raise ValueError('Invalid browser timing or events')
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in m['regions']}:
        raise ValueError('Expected exactly one decision per project region')
    by_id={r['id']:r for r in records}
    for r in records:
        if (r.get('choice') not in ('keep','union','intersection') or type(r.get('reviewed')) is not bool
                or (not r['reviewed'] and r['choice']!='keep')):
            raise ValueError('Invalid/unreviewed decision')
    labels={}
    for name,path in m['candidates'].items():
        with rasterio.open(path) as src:
            labels[name]=src.read(1);profile=src.profile.copy()
    with rasterio.open(project/'write_domain.tif') as src: domain=src.read(1)>0
    initial=labels['center']; result=initial.copy();write=np.zeros_like(domain);logs=[]
    for row in m['regions']:
        r=by_id[row['id']]; x,y,w,h=(row[k] for k in ('x','y','width','height'))
        sl=np.s_[y:y+h,x:x+w];mask=domain[sl]
        if r['choice']!='keep':
            result[sl][mask]=labels[r['choice']][sl][mask];write[sl]|=mask
        logs.append(dict(row,choice=r['choice'],reviewed=r['reviewed'],
                         changed_pixels=int(((result[sl]!=initial[sl])&mask).sum())))
    if np.any((result!=initial)&~write): raise ValueError('Unexpected write outside selected regions')
    verify(project)
    output_dir.mkdir(parents=True)
    save_labels(output_dir/'instance.tif',result,profile)
    from .refine_conflicts import save_mask
    save_mask(output_dir/'selected_write.tif',write,profile)
    save_mask(output_dir/'changed_pixels.tif',result!=initial,profile)
    with (output_dir/'regions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    (output_dir/'decisions.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    manifest=dict(status='complete',project=str(project),project_sha256=digest(project/'project.json'),
                  decisions_sha256=digest(output_dir/'decisions.json'),instance_sha256=digest(output_dir/'instance.tif'),
                  ground_truth_used=False,reviewed_regions=sum(r['reviewed'] for r in records),
                  selected_regions=sum(r['choice']!='keep' for r in records),
                  changed_pixels=int((result!=initial).sum()),
                  selected_roi_pixels=int(write.sum()),regions=logs,
                  self_reported_action_count=len(choices.get('events',[])),
                  timing=choices.get('session_wall_seconds',[]),
                  warning='Human-assisted development experiment; browser wall time includes pauses; baseline provenance unverified')
    (output_dir/'selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    text=(f"Human selection applied; GT not read.\nReviewed regions={manifest['reviewed_regions']}/{len(records)}\n"
          f"Selected regions={manifest['selected_regions']}; selected ROI pixels={manifest['selected_roi_pixels']}\n"
          f"Changed pixels={manifest['changed_pixels']}; self-reported actions={len(events)}\n"
          f"Self-reported browser session seconds={sum(timing):.2f} (includes pauses; not verified active work)\n"
          "instance.tif is a georeferenced processed crop, not a full-image export.\n")
    (output_dir/'summary.txt').write_text(text,encoding='utf-8')
    print(f"Applied: {manifest['selected_regions']} selected regions; {manifest['changed_pixels']} changed pixels")
    return manifest


def evaluate(result_dir,gt_path,output_dir):
    result_dir,gt_path,output_dir=map(lambda p:Path(p).resolve(),(result_dir,gt_path,output_dir))
    if output_dir.exists(): raise FileExistsError(f'Use a new directory: {output_dir}')
    selection=read(result_dir/'selection.json')
    project,m=verify(selection['project'])
    if selection['status']!='complete' or digest(project/'project.json')!=selection['project_sha256']:
        raise ValueError('Selection project changed')
    for file,key in (('decisions.json','decisions_sha256'),('instance.tif','instance_sha256')):
        if digest(result_dir/file)!=selection[key]: raise ValueError('Committed selection changed')
    params=m['parameters'];x0,y0,x1,y1=m['crop_xyxy'];window=Window(x0,y0,x1-x0,y1-y0)
    with rasterio.open(params['image']) as image,rasterio.open(gt_path) as gt:
        aligned(image,gt,params.get('allow_crs_mismatch',False))
        if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
            raise ValueError('GT must contain integer IDs with NoData=None/0')
        truth=gt.read(1,window=window)
        if np.any(truth<0): raise ValueError('Negative GT IDs')
    with rasterio.open(project/'write_domain.tif') as src: domain=src.read(1)>0
    paths=dict(m['candidates'],manual=str(result_dir/'instance.tif'))
    rows=[];entries={}
    for method,path in paths.items():
        with rasterio.open(path) as src: labels=src.read(1)
        counts=roi_counts(labels,truth,domain,params.get('boundary_tolerance',2))
        from .diagnose_regions import transitions
        with rasterio.open(m['candidates']['center']) as src: before=src.read(1)
        entries[method]=dict(scores(counts),counts=counts,transitions_from_center=transitions(before,labels,truth,domain))
        rows.append(dict(method=method,**scores(counts)))
    output_dir.mkdir(parents=True)
    with (output_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(evaluation=entries,selection=selection,gt_sha256=digest(gt_path),
                selection_manifest_sha256=digest(result_dir/'selection.json'),
                warning='Human-assisted developer ROI, fixed IDs, independence unverified; never select candidates from these scores')
    (output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[report['warning']]
    for r in rows:
        lines.append(f"{r['method']}: IoU={r['mask_iou']:.6f}, boundary_F1={r['boundary_f1']:.6f}, "
                     f"instance_F1={r['instance_f1']:.6f}, PQ={r['pq']:.6f}")
    lines.append(f"Reviewed regions={selection['reviewed_regions']}; selected={selection['selected_regions']}; "
                 f"changed pixels={selection['changed_pixels']}; selected ROI pixels={selection['selected_roi_pixels']}")
    (output_dir/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--region-size',type=int,default=256);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--project-dir',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare': prepare(a.run_dir,a.output_dir,a.region_size)
    elif a.command=='apply': apply(a.project_dir,a.decisions,a.output_dir)
    else: evaluate(a.result_dir,a.gt,a.output_dir)


if __name__=='__main__': main()
