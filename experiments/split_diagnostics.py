"""Ground-truth-free images and candidate rejection reports for positive-point refinement."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from .report_regions import overlay,save_png


REASONS={
    'low_score':'质量分数低于阈值',
    'non_finite_score':'质量分数非有限值',
    'non_finite_mask':'掩膜包含非有限值',
    'own_point_missing':'限制范围后，候选未覆盖自己的正点',
    'contains_other_points':'候选连通区域同时包含其他田块正点',
    'too_small':'自己的连通区域小于最小面积',
}
STATUSES={
    'protected':'已有其他实例，未调用SAM，保持保护',
    'no_eligible_candidate':'所有候选均未通过当前规则',
    'selected_before_overlap':'已选中，尚未完成重叠处理',
    'lost_seed_after_overlap':'重叠处理后丢失正点',
    'too_small_after_overlap':'重叠处理后面积过小',
    'retained':'已保留到最终候选',
}


def image_sink(directory,parent_id,rgb):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def save(record,raw,supported,component):
        stem=f'instance_{parent_id}_point_{record["seed"]+1}_candidate_{record["candidate"]+1}'
        path=directory/(stem+'.npz')
        np.savez_compressed(path,raw=raw,supported=supported,component=component)
        record['mask_archive']='diagnostics/'+path.name
        record['images']={}
        for name,mask in (('raw',raw),('supported',supported),('component',component)):
            view=rgb.copy();view[mask]=(.7*view[mask]+.3*np.array([20,170,255])).astype(np.uint8)
            view=overlay(view,mask.astype(np.uint32),np.ones_like(mask))
            # Preserve full-resolution binary masks in NPZ; reduce only display PNG size.
            step=max(1,int(np.ceil(max(mask.shape)/1000)))
            file=directory/(stem+'_'+name+'.png');save_png(file,view[::step,::step])
            record['images'][name]='diagnostics/'+file.name
    return save


def panel(path,points,width,height):
    circles=''
    for i,(x,y) in enumerate(points,1):
        circles+=f'<circle cx="{x}" cy="{y}" r="7" fill="red" stroke="white"/><text x="{x+10}" y="{y}" fill="white" stroke="black" stroke-width=".5" font-size="24">{i}</text>'
    return (f'<svg viewBox="0 0 {width} {height}" style="width:100%"><image href="{html.escape(path)}" '
            f'width="{width}" height="{height}"/>{circles}</svg>')


def write_report(output,proposals,min_area,min_score):
    output=Path(output);cards=[];rows=[]
    for proposal in proposals:
        lab=proposal['id'];x0,y0,x1,y1=proposal['xyxy'];points=proposal['points']
        cards.append(f'<h2>源实例 {lab}</h2>')
        for seed in proposal['detail'].get('diagnostics',[]):
            status=STATUSES.get(seed['status'],seed['status'])
            cards.append(f'<h3>点{seed["seed"]+1}：{html.escape(status)}</h3>')
            cards.append(f'<p>局部坐标{seed["point"]}，原图坐标({x0+seed["point"][0]}, {y0+seed["point"][1]})。'
                         f'提示框：{html.escape(str(seed["box"]))}。红点编号对应原标注。</p>')
            if not seed['candidates']: cards.append('<p>无SAM候选：该点已有其他实例ID，未进行推理。</p>')
            for candidate in seed['candidates']:
                reasons=[REASONS.get(code,code) for code in candidate['rejection_codes']]
                decision='；'.join(reasons) or ('通过，质量分数最高，进入重叠处理' if candidate['selected'] else '通过，但质量分数不是最高，未选用')
                score='非有限值' if candidate['score'] is None else f'{candidate["score"]:.6f}'
                cards.append(f'<article><h4>候选{candidate["candidate"]+1} · 质量分数 {score}</h4>'
                    f'<p><b>{html.escape(decision)}</b></p>'
                    f'<p>原始前景面积 {candidate["raw_pixels"]}，限制后 {candidate["supported_pixels"]}，'
                    f'自己的连通区域 {candidate["component_pixels"]}，被保护范围移除 {candidate["clipped_pixels"]}。'
                    f'原始掩膜覆盖正点 {candidate["raw_contains_points"]}，连通区域包含其他正点 {candidate["component_contains_other_points"]}。'
                    f'自己的正点在原始掩膜内：{candidate["own_point_in_raw"]}；连通区域触及上下文边缘：{candidate["touches_context_edge"]}。</p>')
                images=candidate.get('images',{})
                cards.append('<div class="panels">'+''.join('<figure>'+panel(images[name],points,x1-x0,y1-y0)+f'<figcaption>{label}</figcaption></figure>'
                    for name,label in (('raw','SAM原始掩膜（仅阈值化，未限制修改范围）'),('supported','保护其他实例及无效影像区域后'),('component','与当前正点连通的区域')) if name in images)+'</div></article>')
                rows.append(dict(instance_id=lab,point=seed['seed']+1,candidate=candidate['candidate']+1,
                    final_point_status=seed['status'],score=candidate['score'],eligible=candidate['eligible'],
                    selected_before_overlap=candidate['selected'],raw_pixels=candidate['raw_pixels'],
                    supported_pixels=candidate['supported_pixels'],component_pixels=candidate['component_pixels'],
                    other_points=';'.join(map(str,candidate['component_contains_other_points'])),
                    rejection_codes=';'.join(candidate['rejection_codes'])))
    page=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAM逐点候选诊断</title>'
          '<style>body{font-family:sans-serif;margin:20px}.panels{display:flex;flex-wrap:wrap}figure{width:31%;min-width:280px;margin:8px}article{border:1px solid #bbb;padding:12px;margin-bottom:12px}p{line-height:1.6}</style>'
          '<h1>SAM逐点原始候选与拒绝原因</h1>'
          f'<p>固定质量阈值 {min_score}，最小面积 {min_area} 像素。本页不读取GT、不改变筛选条件。'
          '蓝色填充为候选范围，黄色为其边界，红点为人工标注。分数是SAM预测质量，不是与GT计算的真实IoU。'
          '同一候选可能同时不满足多条规则；“包含其他点”由程序事后检查，并非把这些点输入SAM作为负点。'
          '原始掩膜可延伸到不允许修改的区域，仅用于诊断，不能从本页直接接受或导出。</p>'
          '<p><a href="review.html">返回候选确认页</a></p>'+''.join(cards)+'</html>')
    (output/'diagnostics.html').write_text(page,encoding='utf-8')
    (output/'diagnostics.json').write_text(json.dumps(dict(min_area=min_area,min_score=min_score,
        ground_truth_used=False,instances=[dict(id=p['id'],points=p['points'],diagnostics=p['detail'].get('diagnostics',[])) for p in proposals]),indent=2),encoding='utf-8')
    with (output/'diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        columns=list(rows[0]) if rows else ['instance_id','point','candidate','rejection_codes']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
