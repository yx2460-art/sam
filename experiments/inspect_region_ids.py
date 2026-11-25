"""Offline click-to-inspect current regional instance IDs; no GT or edits."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .region_workflow import load,write_json
from .report_regions import overlay,save_png
from .run_tiles import digest
from .split_instances import sealed,rgb_read


def prepare(result,output):
    result=Path(result).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new inspection directory')
    report=sealed(result/'region_result.json');project=Path(report['interaction_project'])
    workflow=sealed(project/'workflow.json');experiment,m=load(workflow['experiment'])
    if report['experiment_sha256']!=digest(experiment/'experiment.json'):raise ValueError('Result belongs to another region')
    source=result/'region_result.tif'
    with rasterio.open(source) as src,rasterio.open(m['image']) as image,rasterio.open(m['mask']) as roi:
        data=src.read(1);mask=roi.read(1)>0
        rgb=rgb_read(image,Window(0,0,image.width,image.height),[1,2,3],None)
    ownership=[];allowed=[]
    for line,valid in zip(data,mask):
        starts=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1]
        ownership.append([[int(x),int(line[x])] for x in starts])
        delta=np.diff(np.r_[0,valid.astype(np.int8),0])
        allowed.append([[int(a),int(b)] for a,b in zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1))])
    ids,counts=np.unique(data[mask],return_counts=True)
    output.mkdir(parents=True);save_png(output/'current.png',overlay(rgb,data,mask));save_png(output/'rgb.png',rgb)
    config=json.dumps(dict(ownership=ownership,allowed=allowed,offset=m['context_xyxy'][:2],
        experiment=str(experiment),scores=workflow['fusion_scores'],method=report['start_method'],result=str(result)))
    script='const c='+config+''',canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),img=new Image();
let last=null;function draw(){ctx.drawImage(img,0,0);if(last){ctx.beginPath();ctx.arc(last[0],last[1],7,0,Math.PI*2);ctx.strokeStyle='red';ctx.lineWidth=3;ctx.stroke();}}
img.onload=draw;img.src='current.png';document.getElementById('raw').onchange=e=>{img.src=e.target.checked?'rgb.png':'current.png';};
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=Math.floor((e.clientY-r.top)*canvas.height/r.height);
if(x<0||y<0||x>=canvas.width||y>=canvas.height)return;let id=0;for(const [start,value] of c.ownership[y]){if(start>x)break;id=value;}
const valid=c.allowed[y].some(([a,b])=>x>=a&&x<b);last=[x,y];draw();
document.getElementById('picked').textContent='当前实例ID：'+id+'；局部坐标 ('+x+', '+y+')；原图坐标 ('+(x+c.offset[0])+', '+(y+c.offset[1])+')；'+(valid?'在固定评分／回写区域内':'在变暗上下文内，本轮不能修改');
document.getElementById('target').textContent=valid?(id>0?'下一轮使用 --target-id '+id+'；请在怀疑粘连的多个真实田块内部逐一点选，检查是否共用ID。':'这里是预测背景，可做漏检补充；不是一个已有前景实例。'):'该位置需要另建固定区域实验，不能混入当前评分。';};'''
    page='<!doctype html><meta charset="utf-8"><title>当前区域实例ID查看</title><style>body{font-family:sans-serif;margin:20px}canvas{max-width:97vw;height:auto}header{position:sticky;top:0;background:white;padding:10px}p{line-height:1.6}</style>'
    page+='<header><h1>查看当前结果的实例ID</h1><p id="picked">点击上方待处理田块内部</p><p id="target"></p><label><input type="checkbox" id="raw">只看原影像</label></header>'
    page+='<p>黄色为当前预测实例边界；点击各真实田块内部比较ID。相同ID可能粘连，也可能是同一ID的断开区域，需结合原图。不同ID已在标签上分开，不必仅因颜色相近再次拆分。本页不读取GT，不运行SAM，不修改结果。</p>'
    page+=f'<canvas id="map" width="{data.shape[1]}" height="{data.shape[0]}"></canvas><script>'+script+'</script>'
    (output/'index.html').write_text(page,encoding='utf-8')
    plan=dict(status='ready',source_result=str(result),experiment=str(experiment),
        roi_id_pixels={str(int(i)):int(n) for i,n in zip(ids,counts)},ground_truth_used=False,
        input_sha256={str(p):digest(p) for p in [source,result/'region_result.json',project/'workflow.json',experiment/'experiment.json',Path(m['mask']),Path(m['image'])]})
    write_json(output/'inspection.json',plan)
    print(f'Open {output}/index.html; click parcel interiors to inspect IDs; no edits, no GT')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();prepare(a.result_dir,a.output_dir)


if __name__=='__main__':main()
