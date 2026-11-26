"""Review closed background components surrounded by one parcel; no automatic filling or GT."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .manual_fusion import read
from .raster import save_labels
from .refine_conflicts import save_mask
from .region_workflow import load,write_json
from .report_regions import save_png
from .run_tiles import digest
from .split_instances import sealed,rgb_read


def candidates(data,roi,target_id):
    if type(target_id) is not int or target_id<=0 or not np.any((data==target_id)&roi):raise ValueError('Positive target ID absent from scoring ROI')
    background,count=ndimage.label(data==0,structure=np.ones((3,3)))
    accepted=np.zeros_like(data,dtype=np.uint32);rows=[]
    skipped=dict(context_edge=0,outside_roi=0,other_surrounding_ids=0)
    h,w=data.shape
    for index,box in enumerate(ndimage.find_objects(background),1):
        if box is None:continue
        ys,xs=box
        local=(slice(max(0,ys.start-1),min(h,ys.stop+1)),slice(max(0,xs.start-1),min(w,xs.stop+1)))
        component=background[local]==index
        ring=ndimage.binary_dilation(component,structure=np.ones((3,3)))&~component
        neighbors=set(np.unique(data[local][ring]).tolist())-{0}
        if target_id not in neighbors:continue
        if ys.start==0 or xs.start==0 or ys.stop==h or xs.stop==w:skipped['context_edge']+=1;continue
        if np.any(component&~roi[local]):skipped['outside_roi']+=1;continue
        if neighbors!={target_id}:skipped['other_surrounding_ids']+=1;continue
        accepted[local][component]=index
        rows.append(dict(id=index,pixels=int(component.sum()),local_xyxy=[xs.start,ys.start,xs.stop,ys.stop]))
    return accepted,rows,skipped


def prepare(result,target_id,output):
    result=Path(result).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new hole review directory')
    prior=sealed(result/'region_result.json');old_project=Path(prior['interaction_project']);old=sealed(old_project/'workflow.json')
    experiment,m=load(old['experiment'])
    if prior['experiment_sha256']!=digest(experiment/'experiment.json'):raise ValueError('Region mismatch')
    source=result/'region_result.tif'
    with rasterio.open(source) as src,rasterio.open(m['mask']) as roi,rasterio.open(m['image']) as image:
        data=src.read(1);mask=roi.read(1)>0;profile=src.profile;rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
    labels,rows,skipped=candidates(data,mask,target_id)
    output.mkdir(parents=True);save_labels(output/'holes.tif',labels,profile)
    view=rgb.copy();view[data==target_id]=(.65*view[data==target_id]+.35*np.array([0,200,255])).astype(np.uint8)
    view[labels>0]=[255,40,40];view[~mask]=(view[~mask]*.25).astype(np.uint8);save_png(output/'overview.png',view)
    cards=[]
    for row in rows:
        index=row['id'];a,b,c,d=row['local_xyxy'];offset=m['context_xyxy'];row['global_xyxy']=[a+offset[0],b+offset[1],c+offset[0],d+offset[1]]
        a=max(0,a-24);b=max(0,b-24);c=min(data.shape[1],c+24);d=min(data.shape[0],d+24)
        local_view=rgb[b:d,a:c].copy();target=data[b:d,a:c]==target_id
        local_view[target]=(.65*local_view[target]+.35*np.array([0,200,255])).astype(np.uint8)
        local_view[labels[b:d,a:c]==index]=[255,40,40]
        save_png(output/f'{index}_raw.png',rgb[b:d,a:c]);save_png(output/f'{index}_candidate.png',local_view)
        cards.append(f'<section><h2>孔洞 {index}：{row["pixels"]}像素</h2><p>原图范围 {row["global_xyxy"]}。左：原图；右：本孔洞红色，目标田块半透明青色。</p><div class="pair"><img src="{index}_raw.png"><img src="{index}_candidate.png"></div><label><input type="checkbox" id="h{index}">确认此孔洞属于田块，填入ID {target_id}</label><p>真实田埂、非耕作区域或归属不明请保持不勾选。一次确认作用于整个红色孔洞。</p></section>')
    inputs=[source,result/'region_result.json',old_project/'workflow.json',experiment/'experiment.json',Path(m['mask']),Path(m['image']),output/'holes.tif']
    plan=dict(source_result=str(result),source=str(source),experiment=str(experiment),target_id=target_id,mask=m['mask'],holes=rows,skipped=skipped,
        connectivity=8,ground_truth_used=False,input_sha256={str(p):digest(p) for p in inputs})
    write_json(output/'plan.json',plan)
    workflow=dict(old,previous_result=str(result),target_id=target_id,reason='Human-reviewed closed background hole completion',operation='reviewed-hole-fill',
        input_sha256={str(p):digest(p) for p in [output/'plan.json',old_project/'workflow.json',experiment/'experiment.json']})
    write_json(output/'workflow.json',workflow)
    config=json.dumps(dict(hash=digest(output/'plan.json'),ids=[row['id'] for row in rows]))
    script='const c='+config+''';function download(){const accepted_ids=c.ids.filter(id=>document.getElementById('h'+id).checked);if(!accepted_ids.length){alert('没有确认填补孔洞，所有背景保持原状');return;}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({plan_sha256:c.hash,accepted_ids},null,2)],{type:'application/json'}));a.download='hole_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}'''
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>人工确认孔洞填补</title><style>body{font-family:sans-serif;margin:20px}.pair{display:flex;gap:1%}.pair img{width:49%;object-fit:contain}img{max-width:100%}section{padding:15px;border-bottom:1px solid #aaa}</style>'
        f'<h1>ID {target_id}：封闭背景孔洞确认</h1><p>候选 {len(rows)} 个。只提取被此ID独自包围、完全在固定ROI内的8连通背景；外连缺口、裁块边缘背景及涉及其他实例的背景不填。未读取GT，不自动应用。候选也可能是真实非耕作区，请根据原图判断。</p><button onclick="download()">导出hole_decisions.json</button><img src="overview.png">'+(''.join(cards) if rows else '<p>没有符合条件的封闭孔洞。若缺口连到外部背景，应另做局部修正，不能直接填洞。</p>')+'<script>'+script+'</script>',encoding='utf-8')
    print(f'Open {output}/index.html; closed holes={len(rows)}; skipped={skipped}; no GT, no automatic filling')
    return plan


def apply(project,decisions,output):
    project=Path(project).resolve();output=Path(output).resolve();p=sealed(project/'plan.json');w=sealed(project/'workflow.json');a=read(decisions)
    if output.exists():raise FileExistsError('Use a new result directory')
    if a.get('plan_sha256')!=digest(project/'plan.json'):raise ValueError('Hole decisions belong to another project')
    ids=a.get('accepted_ids',[]);valid={row['id'] for row in p['holes']}
    if not ids or any(type(i) is not int for i in ids) or len(set(ids))!=len(ids) or not set(ids)<=valid:raise ValueError('Invalid or empty accepted hole IDs')
    with rasterio.open(p['source']) as src,rasterio.open(p['mask']) as roi,rasterio.open(project/'holes.tif') as holes:
        before=src.read(1);mask=roi.read(1)>0;labels=holes.read(1)
        expected,_,_=candidates(before,mask,p['target_id'])
        if not np.array_equal(expected,labels):raise ValueError('Hole candidates changed')
        selected=np.isin(labels,ids)
        if np.any(selected&(~mask|(before!=0))):raise ValueError('Hole fill outside background/ROI')
        after=before.copy();after[selected]=p['target_id'];output.mkdir(parents=True)
        save_labels(output/'region_result.tif',after,src.profile);save_mask(output/'changed_pixels.tif',selected,src.profile)
    prior=sealed(Path(p['source_result'])/'region_result.json');experiment,m=load(p['experiment'])
    report=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),interaction_project=str(project),start_method=prior['start_method'],
        reason=w['reason'],previous_result=p['source_result'],operation='reviewed-hole-fill',write_policy='reviewed-background-hole-completion',interaction_round=prior.get('interaction_round',1),
        changed_pixels=int(selected.sum()),added_foreground_pixels=int(selected.sum()),removed_foreground_pixels=0,accepted_hole_ids=ids,ground_truth_used_for_edits=False,
        input_sha256={str(path):digest(path) for path in [project/'workflow.json',project/'plan.json',Path(decisions).resolve(),output/'region_result.tif']})
    write_json(output/'region_result.json',report);print(f'Exported {output}/region_result.tif; filled {report["changed_pixels"]} background pixels; other IDs unchanged')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--target-id',type=int,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--project-dir',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.result_dir,a.target_id,a.output_dir)
    else:apply(a.project_dir,a.decisions,a.output_dir)


if __name__=='__main__':main()
