"""Human-reviewed residual reassignment or explicit background cleanup; no GT or SAM."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .diagnose_region import residual_components
from .manual_fusion import read
from .raster import save_labels
from .refine_conflicts import save_mask
from .region_workflow import load,write_json
from .report_regions import save_png
from .run_tiles import digest
from .split_instances import sealed,rgb_read


def prepare(result,output,target_id=2):
    result=Path(result).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new review directory')
    if type(target_id) is not int or target_id<=0:raise ValueError('Positive residual ID required')
    prior=sealed(result/'region_result.json');old_project=Path(prior['interaction_project'])
    old_workflow=sealed(old_project/'workflow.json');experiment,m=load(old_workflow['experiment'])
    if prior['experiment_sha256']!=digest(experiment/'experiment.json'):raise ValueError('Result belongs to another experiment')
    source=result/'region_result.tif'
    with rasterio.open(source) as src,rasterio.open(m['mask']) as roi,rasterio.open(m['image']) as image:
        data=src.read(1);mask=roi.read(1)>0;profile=src.profile
        rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
    components,rows=residual_components(data,mask,target_id,m['context_xyxy'][:2])
    if not rows:raise ValueError('Residual ID absent from scoring region')
    output.mkdir(parents=True);save_labels(output/'components.tif',components,profile)
    colors=['#00c8ff','#80e040','#df70ff','#ffad33','#ffffff','#e07090']
    cards=[]
    for row in rows:
        index=row['component'];targets=sorted(int(i) for i in row['adjacent_ids'] if int(i) not in (0,target_id))
        row['allowed_targets']=targets
        a,b,c,d=row['local_xyxy'];a=max(0,a-32);b=max(0,b-32);c=min(mask.shape[1],c+32);d=min(mask.shape[0],d+32)
        raw=rgb[b:d,a:c].copy();view=raw.copy();local=data[b:d,a:c]
        legend=[]
        for n,lab in enumerate(targets):
            color=colors[n%len(colors)];value=np.array([int(color[k:k+2],16) for k in (1,3,5)])
            selected=(local==lab)&mask[b:d,a:c];view[selected]=(.5*view[selected]+.5*value).astype(np.uint8)
            legend.append(f'<span style="border-bottom:5px solid {color}">ID {lab}</span>')
        view[components[b:d,a:c]==index]=[255,40,40]
        save_png(output/f'{index}_raw.png',raw);save_png(output/f'{index}_review.png',view)
        options='<option value="keep">保留原ID，不修改</option><option value="background">确认为背景／田埂（设为ID 0）</option>'+''.join(f'<option value="{lab}">并入ID {lab}</option>' for lab in targets)
        cards.append(f'<section><h2>区域 {index}：{row["pixels"]}像素</h2><p>原图范围 {row["global_xyxy"]}；接触ROI边缘：{row["touches_roi_edge"]}。</p>'
            f'<div class="pair"><img src="{index}_raw.png"><img src="{index}_review.png"></div><p>左：原图；右：本区域红色，相邻实例半透明着色。'+ '；'.join(legend)+
            f'</p><label>归属选择 <select id="c{index}">{options}</select></label><p>每项操作作用于本编号全部红色像素。只有整个区域都应为背景时才选ID 0；部分田块、部分田埂的混合区域先保留。有歧义保持原ID。</p></section>')
    inputs=[result/'region_result.json',old_project/'workflow.json',experiment/'experiment.json',source,Path(m['mask']),Path(m['image']),output/'components.tif']
    plan=dict(status='ready',operation='review-residual-merge',source_result=str(result),experiment=str(experiment),target_id=target_id,
        source=str(source),mask=m['mask'],components=rows,allow_background_cleanup=True,gt_used_for_proposals=False,input_sha256={str(p):digest(p) for p in inputs})
    write_json(output/'plan.json',plan)
    workflow=dict(old_workflow,previous_result=str(result),reason='Human-reviewed residual reassignment and optional explicit background cleanup',
        target_id=target_id,operation='residual-merge',input_sha256={str(p):digest(p) for p in [output/'plan.json',old_project/'workflow.json',experiment/'experiment.json']})
    write_json(output/'workflow.json',workflow)
    config=json.dumps(dict(hash=digest(output/'plan.json'),ids=[row['component'] for row in rows]))
    script='const config='+config+''';function download(){const assignments=config.ids.map(component=>{const v=document.getElementById('c'+component).value;return v==='keep'?null:{component,target_id:v==='background'?0:Number(v),action:v==='background'?'background':'merge'};}).filter(Boolean);
if(!assignments.length){alert('没有选择修改区域，默认全部保留');return;}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({plan_sha256:config.hash,assignments},null,2)],{type:'application/json'}));a.download='merge_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
async function restore(input){try{const v=JSON.parse(await input.files[0].text());if(v.plan_sha256!==config.hash||!Array.isArray(v.assignments))throw Error('请选择本页面导出的决策文件');
const checked=[],seen=new Set();for(const r of v.assignments){const select=document.getElementById('c'+r.component),value=r.action==='background'&&r.target_id===0?'background':String(r.target_id);
if(!select||seen.has(r.component)||!Array.from(select.options).some(o=>o.value===value))throw Error('决策区域或目标不匹配');seen.add(r.component);checked.push([select,value]);}
config.ids.forEach(id=>document.getElementById('c'+id).value='keep');checked.forEach(([select,value])=>select.value=value);}catch(e){alert(e.message);}}'''
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>残余人工归属确认</title><style>body{font-family:sans-serif;margin:20px}.pair{display:flex;gap:1%}.pair img{width:49%;object-fit:contain}section{border-bottom:1px solid #aaa;padding:16px}header{position:sticky;top:0;background:white;padding:12px}</style>'
        '<header><b>残余归属与背景清理确认</b> 不显示GT，不自动选择；背景选项会删除对应前景。<button onclick="download()">导出 merge_decisions.json</button><label>恢复本页面已导出的选择<input type="file" accept=".json" onchange="restore(this)"></label></header>'
        '<p>只审查固定评分区域内的残余，ROI裁切可能分断实例。选择必须根据原影像田埂判断；颜色表示当前ID，不是正确性。所有项目默认保留，可只确认部分区域。</p>'
        +''.join(cards)+'<script>'+script+'</script>',encoding='utf-8')
    print(f'Open {output}/index.html; {len(rows)} components; no GT read; defaults keep all')
    return plan


def apply(project,decisions,output):
    project=Path(project).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new result directory')
    plan=sealed(project/'plan.json');workflow=sealed(project/'workflow.json');decision=read(decisions)
    if decision.get('plan_sha256')!=digest(project/'plan.json'):raise ValueError('Decisions belong to another review')
    choices=decision.get('assignments',[]);available={row['component']:row for row in plan['components']};seen=set()
    if not choices:raise ValueError('No residual assignments accepted')
    for row in choices:
        index=row.get('component');target=row.get('target_id')
        if type(index) is not int or type(target) is not int or index in seen or index not in available:
            raise ValueError('Invalid, duplicate, or nonadjacent residual assignment')
        if target==0:
            if row.get('action')!='background' or not plan.get('allow_background_cleanup',False):raise ValueError('Invalid background action: explicit background confirmation required')
        elif target not in available[index]['allowed_targets'] or row.get('action','merge')!='merge':raise ValueError('Invalid merge destination/action')
        seen.add(index)
    experiment,m=load(plan['experiment']);prior=sealed(Path(plan['source_result'])/'region_result.json')
    with rasterio.open(plan['source']) as src,rasterio.open(plan['mask']) as roi,rasterio.open(project/'components.tif') as labels:
        before=src.read(1);data=before.copy();mask=roi.read(1)>0;components=labels.read(1);deleted=np.zeros_like(mask)
        for row in choices:
            selected=components==row['component']
            if np.any(selected&(~mask|(before!=plan['target_id']))):raise ValueError('Component outside residual write scope')
            data[selected]=row['target_id']
            if row['target_id']==0:deleted|=selected
        if np.any((data>0)!=((before>0)&~deleted)) or np.any(data[~mask]!=before[~mask]):raise ValueError('Unexpected foreground or context change')
        output.mkdir(parents=True);save_labels(output/'region_result.tif',data,src.profile);save_mask(output/'changed_pixels.tif',data!=before,src.profile)
    sealed(project/'plan.json')
    report=dict(status='complete',experiment_sha256=digest(experiment/'experiment.json'),interaction_project=str(project),
        start_method=prior['start_method'],reason=workflow['reason'],operation='residual-cleanup' if deleted.any() else 'residual-merge',write_policy='reviewed-background-cleanup' if deleted.any() else 'preserve-foreground-residual-merge',
        previous_result=plan['source_result'],interaction_round=prior.get('interaction_round',1),
        changed_pixels=int((data!=before).sum()),scoring_pixels=int(mask.sum()),assignments=choices,
        added_foreground_pixels=0,removed_foreground_pixels=int(deleted.sum()),merged_pixels=int(((data!=before)&~deleted).sum()),ground_truth_used_for_edits=False,
        input_sha256={str(p):digest(p) for p in [project/'workflow.json',project/'plan.json',Path(decisions).resolve(),output/'region_result.tif']})
    write_json(output/'region_result.json',report);print(f'Exported {output}/region_result.tif; merged={report["merged_pixels"]}; removed foreground={report["removed_foreground_pixels"]}; policy={report["write_policy"]}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--target-id',type=int,default=2);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--project-dir',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.result_dir,a.output_dir,a.target_id)
    else:apply(a.project_dir,a.decisions,a.output_dir)


if __name__=='__main__':main()
