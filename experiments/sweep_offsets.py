"""Prespecified half-stride grid-phase checks for two development configurations."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .evaluate_tiles import roi_counts
from .metrics import scores
from .report_regions import overlay, save_png
from .run_tiles import aligned, digest
from .sweep_strides import PATH_KEYS, read_json
from .tiles import mask_bounds, windows


TARGETS=((256,192,'union'),(256,64,'majority'))
METRICS=('mask_iou','boundary_f1','instance_f1','pq')


def phases(stride):
    half=stride//2
    return [('original',0,0),('horizontal',half,0),('vertical',0,half),('both',half,half)]


def execute(p):
    command=[sys.executable,'-m','experiments.run_tiles']
    for key in (*PATH_KEYS,'output_dir','tile_size','stride','context','fusion','model_type','device',
                'boundary_tolerance','offset_x','offset_y'):
        command.extend(['--'+key.replace('_','-'),str(p[key])])
    command.extend(['--rgb-bands',*map(str,p['rgb_bands'])])
    if p['rgb_range']: command.extend(['--rgb-range',*map(str,p['rgb_range'])])
    if p['allow_crs_mismatch']: command.append('--allow-crs-mismatch')
    subprocess.run(command,check=True)


def artifacts(directory):
    return {name:digest(directory/name) for name in ('manifest.json','instance.tif','count.tif')}


def validate(directory,p,hashes,prompt):
    m=read_json(directory/'manifest.json')
    if m.get('status')!='complete' or m['grid_origin']!=[p['offset_x'],p['offset_y']]:
        raise ValueError(f'Incomplete/incompatible offset run: {directory}')
    if m['fusion']!=p['fusion'] or m['fixed_prompt_parameters']!=prompt:
        raise ValueError('Fusion/prompt protocol changed')
    for key in ('tile_size','stride','context','model_type','device','rgb_bands','rgb_range',
                'allow_crs_mismatch','boundary_tolerance'):
        if m['parameters'][key]!=p[key]: raise ValueError(f'Run setting changed: {key}')
    for key in PATH_KEYS:
        if Path(m['parameters'][key]).resolve()!=Path(p[key]).resolve():
            raise ValueError(f'Run input path changed: {key}')
        if key in m['hashes'] and m['hashes'][key]!=hashes[key]:
            raise ValueError(f'Run input hash changed: {key}')
    return m


def fixed_cases(roi,size=256,count=4,seed=41):
    x0,y0,x1,y1=mask_bounds(roi,0)
    candidates=[]
    for y in range(y0//size*size,y1,size):
        for x in range(x0//size*size,x1,size):
            w,h=min(size,roi.width-x),min(size,roi.height-y)
            window=Window(x,y,w,h)
            raw=roi.read(1,window=window)
            selected=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=window)>0)
            if selected.any(): candidates.append((x,y,w,h))
    rng=np.random.default_rng(seed)
    indices=sorted(rng.choice(len(candidates),min(count,len(candidates)),replace=False).tolist())
    return [candidates[i] for i in indices]


def visuals(output,parameters,gt_path,jobs):
    # Samples depend only on ROI geometry and a fixed seed, never GT/errors.
    with ExitStack() as stack:
        roi=stack.enter_context(rasterio.open(parameters['eval_mask']))
        image=stack.enter_context(rasterio.open(parameters['image']))
        gt=stack.enter_context(rasterio.open(gt_path))
        base=stack.enter_context(rasterio.open(parameters['prediction']))
        chosen=fixed_cases(roi)
        fragments=[]
        records=[]
        zero=[j for j in jobs if j['phase']=='original']
        for index,(x,y,w,h) in enumerate(chosen):
            win=Window(x,y,w,h)
            rgb=image.read(parameters['rgb_bands'],window=win)
            valid=np.all(image.read_masks(parameters['rgb_bands'],window=win)>0,axis=0)
            if parameters['rgb_range']:
                low,high=parameters['rgb_range']
                rgb=np.clip((rgb.astype(np.float32)-low)*255/(high-low),0,255)
            elif rgb.dtype!=np.uint8:
                raise ValueError('Non-uint8 RGB requires recorded rgb_range')
            rgb=np.moveaxis(rgb,0,-1).astype(np.uint8);rgb[~valid]=0
            raw=roi.read(1,window=win)
            domain=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
            truth=gt.read(1,window=win)
            panels=[('RGB',rgb),('GT boundary',overlay(rgb,truth,domain))]
            labels=[('BsiNet',base.read(1,window=win))]
            for job in zero:
                m=read_json(Path(job['run_dir'])/'manifest.json')
                cx,cy,_,_=m['crop_xyxy']
                with rasterio.open(Path(job['run_dir'])/'instance.tif') as src:
                    data=src.read(1,window=Window(x-cx,y-cy,w,h),boundless=True,fill_value=0)
                labels.append((job['tag'],data))
            for name,data in labels:
                error=rgb.copy()
                error[~domain]=(error[~domain]*.2).astype(np.uint8)
                fp=(data>0)&(truth==0)&domain
                fn=(data==0)&(truth>0)&domain
                error[fp]=[0,100,255];error[fn]=[255,0,0]
                panels.extend([(name+' boundary',overlay(rgb,data,domain)),(name+' FP blue / FN red',error)])
            cards=[]
            for number,(label,picture) in enumerate(panels):
                filename=f'case_{index:02d}_{number}.png'
                save_png(output/filename,picture)
                cards.append(f'<figure><img src="{filename}" width="256"><figcaption>{label}</figcaption></figure>')
            fragments.append(f'<h2>Fixed ROI example x={x}, y={y}</h2><div class="panels">'+''.join(cards)+'</div>')
            records.append(dict(x=x,y=y,width=w,height=h))
        (output/'cases.json').write_text(json.dumps(dict(seed=41,size=256,
            selection='uniform sample without replacement from original-image-origin ROI-intersecting tiles; no GT selection',cases=records),indent=2),encoding='utf-8')
        return ''.join(fragments)


def sweep(fusion_dir,output_dir,plan_only=False,runner=None):
    fusion_dir,output_dir=Path(fusion_dir).resolve(),Path(output_dir).resolve()
    source_path=fusion_dir/'fusion_sweep.json'
    fusion=read_json(source_path)
    if fusion.get('status')!='complete': raise ValueError('Fusion sweep must be complete')
    dense_path=Path(fusion['signature']['source_sweep'])
    if digest(dense_path)!=fusion['signature']['source_sweep_sha256']:
        raise ValueError('Dense sweep changed')
    dense=read_json(dense_path)
    sig=dense['signature'];parameters=sig['parameters'];gt_path=Path(sig['gt'])
    for key in PATH_KEYS:
        if digest(parameters[key])!=sig['hashes'][key]: raise ValueError(f'Changed input: {key}')
    if digest(gt_path)!=sig['hashes']['gt']: raise ValueError('GT changed')
    for protected in (fusion_dir,dense_path.parent):
        if output_dir==protected or protected in output_dir.parents:
            raise ValueError('Output must be outside source sweeps')
    signature=dict(fusion_sweep=str(source_path),fusion_sha256=digest(source_path),
                   dense_sha256=digest(dense_path),hashes=sig['hashes'],targets=TARGETS)
    signature=json.loads(json.dumps(signature))
    state_path=output_dir/'offset_sweep.json'
    if output_dir.exists():
        if not state_path.is_file() or read_json(state_path)['signature']!=signature:
            raise ValueError('Existing output is not a matching offset sweep')
        state=read_json(state_path)
    else:
        state=dict(signature=signature,jobs=[],status='planning',
                   notes='Post-selection grid sensitivity check on development ROI; four deterministic phases are not independent statistical replicates. No independent-test or significance claim.')
    old={j['name']:j for j in state['jobs']}
    jobs=[]
    source_jobs={(j['tile_size'],j['stride']):j for j in fusion['jobs']}
    with rasterio.open(parameters['image']) as image,rasterio.open(parameters['eval_mask']) as roi:
        bounds=mask_bounds(roi,parameters['context'])
        for tile,stride,method in TARGETS:
            source=source_jobs[(tile,stride)]
            tag=f't{tile}_s{stride}_{method}'
            for phase,dx,dy in phases(stride):
                name=tag+'_'+phase
                p=dict(parameters,tile_size=tile,stride=stride,fusion=method,offset_x=dx,offset_y=dy)
                directory=Path(source['output_dir'])/method if phase=='original' else output_dir/'runs'/name
                p['output_dir']=str(directory)
                if phase=='original':
                    for filename in ('manifest.json','instance.tif','count.tif'):
                        if digest(directory/filename)!=source['output_sha256'][method+'/'+filename]:
                            raise ValueError('Zero-offset fusion source changed')
                if output_dir==directory or directory in output_dir.parents:
                    raise ValueError('Output is inside an input run')
                complete=directory.exists()
                if complete:
                    manifest_path=directory/'manifest.json'
                    if not manifest_path.is_file() or read_json(manifest_path).get('status')!='complete':
                        raise ValueError(f'Partial run cannot be overwritten: {directory}. Preserve/relocate this explicit run before retrying, or use a new sweep directory.')
                    validate(directory,p,sig['hashes'],sig['prompt'])
                    current=artifacts(directory)
                    if name in old and old[name].get('artifact_sha256') and current!=old[name]['artifact_sha256']:
                        raise ValueError('Previously recorded offset artifact changed')
                wins=list(windows(image.width,image.height,tile,stride,bounds,dx,dy))
                pixels=(max(w.col_off+w.width for w in wins)-min(w.col_off for w in wins))*(max(w.row_off+w.height for w in wins)-min(w.row_off for w in wins))
                job=dict(name=name,tag=tag,phase=phase,tile_size=tile,stride=stride,method=method,
                         dx=dx,dy=dy,run_dir=str(directory),parameters=p,status='complete' if complete else 'planned',
                         windows=len(wins),scratch_bytes=int(pixels*13))
                if complete: job['artifact_sha256']=current
                jobs.append(job)
    output_dir.mkdir(parents=True,exist_ok=True)
    state.update(status='planned',jobs=jobs)
    def save(): state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    pending=[j for j in jobs if j['status']!='complete']
    print(f'{len(jobs)} grid configurations; {len(pending)} need SAM inference; zero-offset outputs reused.',flush=True)
    print(f"New scratch maps only={sum(j['scratch_bytes'] for j in pending)/2**30:.2f} GiB; free disk={shutil.disk_usage(output_dir).free/2**30:.2f} GiB; tiles/results extra.",flush=True)
    for j in jobs: print(f"{j['name']}: offset=({j['dx']},{j['dy']}), windows={j['windows']}, {j['status']}",flush=True)
    if plan_only: return state
    try:
        state['status']='running';save()
        for job in jobs:
            if job['status']!='complete':
                (runner or execute)(job['parameters'])
                validate(Path(job['run_dir']),job['parameters'],sig['hashes'],sig['prompt'])
                job.update(status='complete',artifact_sha256=artifacts(Path(job['run_dir'])));save()
        rows=[]
        with ExitStack() as stack:
            image=stack.enter_context(rasterio.open(parameters['image']))
            base=stack.enter_context(rasterio.open(parameters['prediction']))
            roi=stack.enter_context(rasterio.open(parameters['eval_mask']))
            gt=stack.enter_context(rasterio.open(gt_path))
            allow=parameters['allow_crs_mismatch']
            for src in (base,roi,gt): aligned(image,src,allow)
            if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
                raise ValueError('GT requires integer labels and NoData=None/0')
            total_roi=0
            for _,win in roi.block_windows(1):
                raw=roi.read(1,window=win)
                total_roi+=int(((raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)).sum())
            for job in jobs:
                directory=Path(job['run_dir'])
                if artifacts(directory)!=job['artifact_sha256']: raise ValueError('Offset artifact changed')
                m=validate(directory,job['parameters'],sig['hashes'],sig['prompt'])
                if m['processed_windows']!=job['windows']: raise ValueError('Planned window count differs')
                x0,y0,x1,y1=m['crop_xyxy'];win=Window(x0,y0,x1-x0,y1-y0)
                raw=roi.read(1,window=win)
                mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=win)>0)
                if int(mask.sum())!=total_roi: raise ValueError('Crop omits scoring pixels')
                target=gt.read(1,window=win)
                if np.any(target<0): raise ValueError('Negative GT labels')
                for name in ('instance','count'):
                    with rasterio.open(directory/f'{name}.tif') as src:
                        if src.shape!=mask.shape or not src.transform.almost_equals(image.window_transform(win)):
                            raise ValueError('Output crop grid mismatch')
                        data=src.read(1)
                    if name=='instance': prediction=data
                    elif np.any(mask&(data==0)): raise ValueError('Unobserved ROI pixels')
                baseline=scores(roi_counts(base.read(1,window=win),target,mask,parameters['boundary_tolerance']))
                if baseline!=dense['baseline']: raise ValueError('Baseline changes across phases')
                result=scores(roi_counts(prediction,target,mask,parameters['boundary_tolerance']))
                if job['phase']=='original':
                    prior=next(r for r in fusion['rows'] if (r['tile_size'],r['stride'],r['method'])==(job['tile_size'],job['stride'],job['method']))
                    if any(result[k]!=prior[k] for k in result): raise ValueError('Zero phase does not reproduce source metrics')
                rows.append(dict(configuration=job['tag'],phase=job['phase'],offset_x=job['dx'],offset_y=job['dy'],
                    windows=m['processed_windows'],seconds=m.get('source_inference_seconds',m['seconds']),**result))
        with (output_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        variability={}
        for tag in sorted({r['configuration'] for r in rows}):
            variability[tag]={}
            for key in METRICS:
                values=[r[key] for r in rows if r['configuration']==tag]
                variability[tag][key]=dict(mean=float(np.mean(values)),minimum=min(values),
                                          maximum=max(values),range=max(values)-min(values))
        images=visuals(output_dir,parameters,gt_path,jobs)
        charts=[]
        for key in METRICS:
            values=[r[key] for r in rows]+[dense['baseline'][key]]
            lo,hi=min(values)-.002,max(values)+.002
            parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 280"><rect width="680" height="280" fill="white"/><text x="70" y="20">{key}</text><path d="M70 45V225H630" fill="none" stroke="black"/>']
            for i in range(5): parts.append(f'<text x="4" y="{225-i*45}" font-size="11">{lo+(hi-lo)*i/4:.4f}</text>')
            by=225-(dense['baseline'][key]-lo)/(hi-lo)*180
            parts.append(f'<path d="M70 {by}H630" stroke="#555" stroke-dasharray="4 4"/>')
            for color,tag in zip(('#1764b0','#cf5018'),sorted(variability)):
                ordered=[next(r for r in rows if r['configuration']==tag and r['phase']==p) for p in ('original','horizontal','vertical','both')]
                points=' '.join(f'{100+i*165},{225-(r[key]-lo)/(hi-lo)*180}' for i,r in enumerate(ordered))
                parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/><text x="{70 if color=="#1764b0" else 350}" y="275" fill="{color}" font-size="12">{tag}</text>')
            for i,label in enumerate(('original','horizontal','vertical','both')): parts.append(f'<text x="{80+i*165}" y="247" font-size="11">{label}</text>')
            filename=key+'.svg';(output_dir/filename).write_text(''.join(parts)+'</svg>',encoding='utf-8')
            charts.append(f'<img src="{filename}" style="width:680px;max-width:100%">')
        (output_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Grid phase audit</title><style>body{font:15px sans-serif;margin:24px}.panels{display:flex;flex-wrap:wrap}figure{margin:6px}</style><h1>Grid phase audit</h1><p>'+state['notes']+'</p><p>Axes auto-scaled; dashed line=BsiNet. Fixed-seed ROI samples, no GT-based case selection. Yellow=instance boundaries; blue=false foreground; red=missed foreground. Darkened=outside ROI. Visual comparisons below use original phases only.</p>'+''.join(charts)+images,encoding='utf-8')
        if digest(source_path)!=signature['fusion_sha256'] or digest(dense_path)!=signature['dense_sha256']:
            raise ValueError('Source sweep changed')
        for key in PATH_KEYS:
            if digest(parameters[key])!=sig['hashes'][key]: raise ValueError('Input changed')
        if digest(gt_path)!=sig['hashes']['gt']: raise ValueError('GT changed')
        for job in jobs:
            if artifacts(Path(job['run_dir']))!=job['artifact_sha256']:
                raise ValueError('Offset output changed during report generation')
        state.update(status='complete',rows=rows,variability=variability,baseline=dense['baseline'])
        state.pop('error',None);save()
        lines=[state['notes'],'BsiNet: '+', '.join(f'{k}={dense["baseline"][k]:.6f}' for k in METRICS)]
        for r in rows: lines.append(f"{r['configuration']} / {r['phase']}: IoU={r['mask_iou']:.6f}, boundary_F1={r['boundary_f1']:.6f}, instance_F1={r['instance_f1']:.6f}, PQ={r['pq']:.6f}")
        for tag,entry in variability.items():
            lines.append(tag+' across phases: '+', '.join(f'{key} mean={entry[key]["mean"]:.6f}, range={entry[key]["range"]:.6f}' for key in METRICS))
        text='\n'.join(lines)+'\n';(output_dir/'summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
        return state
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fusion-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();sweep(args.fusion_dir,args.output_dir,args.plan_only)


if __name__=='__main__': main()
