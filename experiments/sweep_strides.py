"""Dense stride ablation with validated whole-run reuse and fixed center fusion."""
import argparse
import csv
import html
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
from .run_tiles import aligned, digest
from .tiles import mask_bounds, windows


CONFIGURATIONS=[(size,size*n//8) for size in (256,512) for n in (8,7,6,5,4,3,2)]
PATH_KEYS=('image','prediction','coverage','eval_mask','checkpoint')
FIXED_KEYS=('context','fusion','model_type','device','rgb_bands','rgb_range',
            'allow_crs_mismatch','boundary_tolerance')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compatible(manifest,parameters,hashes,tile,stride,prompt):
    if manifest.get('status')!='complete' or manifest.get('source_run'):
        return False
    p=manifest['parameters']
    if p.get('max_windows')!=0 or p.get('tile_size')!=tile or p.get('stride')!=stride:
        return False
    if manifest.get('fusion')!='center' or manifest.get('fixed_prompt_parameters')!=prompt:
        return False
    if any(p.get(k)!=parameters[k] for k in FIXED_KEYS):
        return False
    if any(Path(p[k]).resolve()!=Path(parameters[k]).resolve() for k in PATH_KEYS):
        return False
    sealed=manifest.get('hashes',{})
    return (all(sealed.get(k)==hashes[k] for k in ('prediction','eval_mask','checkpoint')) and
            all(k not in sealed or sealed[k]==hashes[k] for k in ('image','coverage')))


def validate_run(directory,parameters,hashes,tile,stride,prompt):
    manifest=read_json(directory/'manifest.json')
    if not compatible(manifest,parameters,hashes,tile,stride,prompt):
        raise ValueError(f'Incompatible/incomplete run: {directory}')
    records=[json.loads(line) for line in (directory/'windows.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=manifest['processed_windows'] or len(records)!=manifest['planned_windows']:
        raise ValueError(f'Incomplete window log: {directory}')
    if [r['index'] for r in records]!=list(range(len(records))):
        raise ValueError(f'Invalid window order: {directory}')
    for record in records:
        tile_path=Path(record['tile'])
        if tile_path.is_absolute() or '..' in tile_path.parts or not (directory/tile_path).is_file():
            raise ValueError(f'Missing/invalid cached tile: {directory}')
    for name in ('instance.tif','count.tif'):
        if not (directory/name).is_file():
            raise ValueError(f'Missing {name}: {directory}')
    return manifest,records


def execute(parameters):
    command=[sys.executable,'-m','experiments.run_tiles']
    for key in (*PATH_KEYS,'output_dir','tile_size','stride','context','fusion','model_type','device','boundary_tolerance'):
        command.extend(['--'+key.replace('_','-'),str(parameters[key])])
    command.extend(['--rgb-bands',*map(str,parameters['rgb_bands'])])
    if parameters['rgb_range']:
        command.extend(['--rgb-range',*map(str,parameters['rgb_range'])])
    if parameters['allow_crs_mismatch']:
        command.append('--allow-crs-mismatch')
    # No GT is passed into a child inference run.
    subprocess.run(command,check=True)


def curve(rows,key,title,baseline=None):
    width,height=660,280
    values=[r[key] for r in rows]
    if baseline is not None:
        values.append(baseline)
    low,high=min(values),max(values)
    pad=max((high-low)*0.12,0.002 if key!='seconds' else 1)
    low,high=low-pad,high+pad
    def xy(row):
        return 65+row['overlap_fraction']/0.75*560,230-(row[key]-low)/(high-low)*185
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/><text x="65" y="20">{title}</text>',
            '<path d="M65 40V230H625" fill="none" stroke="black"/>']
    for i in range(5):
        value=low+(high-low)*i/4
        y=230-i*185/4
        pieces.append(f'<text x="2" y="{y:.1f}" font-size="11">{value:.4f}</text>')
    for fraction in (0,.125,.25,.375,.5,.625,.75):
        x=65+fraction/.75*560
        pieces.append(f'<text x="{x-12:.1f}" y="250" font-size="11">{fraction*100:g}%</text>')
    pieces.append('<text x="270" y="275" font-size="12">Overlap per axis</text>')
    if baseline is not None:
        y=230-(baseline-low)/(high-low)*185
        pieces.append(f'<path d="M65 {y:.2f}H625" stroke="#555" stroke-dasharray="5 4"/><text x="68" y="{y-4:.2f}" font-size="10">BsiNet baseline</text>')
    for tile,color in ((256,'#1464c0'),(512,'#c44b17')):
        selected=sorted([r for r in rows if r['tile_size']==tile],key=lambda r:r['overlap_fraction'])
        points=' '.join(f'{xy(r)[0]:.2f},{xy(r)[1]:.2f}' for r in selected)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<text x="{380 if tile==256 else 485}" y="20" fill="{color}">{tile}px</text>')
        for row in selected:
            x,y=xy(row)
            pieces.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"><title>stride={row["stride"]}; value={row[key]:.6f}</title></circle>')
    pieces.append('</svg>')
    return ''.join(pieces)


def sweep(template_run,gt_path,output_dir,plan_only=False,reuse_root=None,runner=None):
    template_run,gt_path,output_dir=map(lambda p:Path(p).resolve(),(template_run,gt_path,output_dir))
    template=read_json(template_run/'manifest.json')
    if template.get('status')!='complete' or template.get('fusion')!='center' or template.get('source_run'):
        raise ValueError('Template must be a complete original center-fusion window run')
    parameters={k:template['parameters'][k] for k in (*PATH_KEYS,*FIXED_KEYS)}
    for key in PATH_KEYS:
        parameters[key]=str(Path(parameters[key]).resolve())
    prompt=template['fixed_prompt_parameters']
    if template['parameters'].get('max_windows')!=0:
        raise ValueError('Smoke template is not allowed')
    # Match the current inference protocol explicitly, rather than silently changing prompts.
    expected=dict(mode='interior',residual_policy='replace',box_pad=2,support_pad=0,min_area=30,
                  min_score=0.8,min_stability=0.9,stability_offset=1,min_support=0.8,min_coverage=0.5,max_overlap=0.05)
    if prompt!=expected:
        raise ValueError('Template prompt protocol differs from current fixed protocol')
    hashes={k:digest(v) for k,v in parameters.items() if k in PATH_KEYS}
    hashes['gt']=digest(gt_path)
    if any(template['hashes'].get(k)!=hashes[k] for k in ('prediction','eval_mask','checkpoint')):
        raise ValueError('Template inputs changed')
    if output_dir==template_run or template_run in output_dir.parents:
        raise ValueError('Output must be outside template run')
    state_path=output_dir/'sweep.json'
    signature=dict(parameters=parameters,hashes=hashes,prompt=prompt,configurations=CONFIGURATIONS,
                   template_run=str(template_run),gt=str(gt_path))
    # Normalize tuples to their on-disk JSON representation for resume comparison.
    signature=json.loads(json.dumps(signature))
    if output_dir.exists():
        if not state_path.is_file() or read_json(state_path)['signature']!=signature:
            raise ValueError('Existing output is not a matching sweep')
        state=read_json(state_path)
    else:
        output_dir.mkdir(parents=True)
        state=dict(status='planning',signature=signature,jobs=[],
                   limitation='Development ROI; training independence unverified; fixed center fusion; no automatic best-setting selection',
                   legacy_reuse='Legacy manifests seal prediction/ROI/checkpoint but not historical image/coverage bytes; current hashes and matching paths are recorded. Wall times include loading and I/O and are not pure GPU times.')
    with rasterio.open(parameters['image']) as image,rasterio.open(parameters['eval_mask']) as roi:
        bounds=mask_bounds(roi,parameters['context'])
        plans={}
        for tile,stride in CONFIGURATIONS:
            wins=list(windows(image.width,image.height,tile,stride,bounds))
            x0=min(w.col_off for w in wins); y0=min(w.row_off for w in wins)
            x1=max(w.col_off+w.width for w in wins); y1=max(w.row_off+w.height for w in wins)
            plans[(tile,stride)]=(len(wins),int((x1-x0)*(y1-y0)*13))
    reuse_root=Path(reuse_root).resolve() if reuse_root else template_run.parent
    candidates=sorted({template_run,*[p.parent for p in reuse_root.glob('*/manifest.json')]})
    previous={(job['tile_size'],job['stride']):job for job in state['jobs']}
    jobs=[]
    for tile,stride in CONFIGURATIONS:
        destination=output_dir/'runs'/f't{tile}_s{stride}'
        if (tile,stride) in previous:
            directory=Path(previous[(tile,stride)]['run_dir'])
        else:
            matches=[p for p in candidates if compatible(read_json(p/'manifest.json'),parameters,hashes,tile,stride,prompt)]
            directory=matches[0] if matches else destination
        complete=(directory/'manifest.json').is_file() and read_json(directory/'manifest.json').get('status')=='complete'
        if directory.exists() and not complete:
            raise ValueError(f'Partial/incompatible run cannot resume within windows: {directory}. Preserve it and use a new sweep directory, or explicitly relocate the partial run before retrying.')
        if complete:
            validate_run(directory,parameters,hashes,tile,stride,prompt)
            for name,expected_hash in previous.get((tile,stride),{}).get('artifact_sha256',{}).items():
                if digest(directory/name)!=expected_hash:
                    raise ValueError(f'Previously summarized artifact changed: {directory/name}')
        count,scratch=plans[(tile,stride)]
        entry=dict(tile_size=tile,stride=stride,overlap_fraction=1-stride/tile,
                         run_dir=str(directory),reused_legacy=directory!=destination,
                         status='complete' if complete else 'planned',planned_windows=count,scratch_bytes=scratch)
        if complete:
            entry['artifact_sha256']={name:digest(directory/name) for name in ('manifest.json','windows.jsonl','instance.tif','count.tif')}
        jobs.append(entry)
    state.update(jobs=jobs,status='planned')
    def save():
        state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    new=[j for j in jobs if j['status']!='complete']
    print(f"Dense sweep: {len(jobs)} configurations; {len(jobs)-len(new)} completed runs reusable; {len(new)} new runs.",flush=True)
    print(f"New scratch maps only: {sum(j['scratch_bytes'] for j in new)/2**30:.2f} GiB; free disk={shutil.disk_usage(output_dir).free/2**30:.2f} GiB. Tiles/outputs need additional space.",flush=True)
    if plan_only:
        for job in jobs:
            print(f"{job['tile_size']} / stride {job['stride']}: {job['planned_windows']} windows, {job['status']}")
        return state
    try:
        state['status']='running';save()
        for job in jobs:
            directory=Path(job['run_dir'])
            if job['status']!='complete':
                p=dict(parameters,output_dir=str(directory),tile_size=job['tile_size'],stride=job['stride'])
                print(f"Starting t{job['tile_size']}_s{job['stride']}",flush=True)
                (runner or execute)(p)
                validate_run(directory,parameters,hashes,job['tile_size'],job['stride'],prompt)
                job['status']='complete';save()
        # GT pixels are not read until every inference configuration has finished.
        rows=[]
        baseline_scores=None
        with ExitStack() as stack:
            image=stack.enter_context(rasterio.open(parameters['image']))
            base=stack.enter_context(rasterio.open(parameters['prediction']))
            gt=stack.enter_context(rasterio.open(gt_path))
            roi=stack.enter_context(rasterio.open(parameters['eval_mask']))
            allow=parameters['allow_crs_mismatch']
            for src in (base,gt,roi): aligned(image,src,allow)
            if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
                raise ValueError('GT must contain integer labels with NoData=None/0')
            full_roi=0
            for _,block in roi.block_windows(1):
                raw=roi.read(1,window=block)
                valid=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=block)>0)
                full_roi+=int(valid.sum())
            for job in jobs:
                directory=Path(job['run_dir'])
                manifest,records=validate_run(directory,parameters,hashes,job['tile_size'],job['stride'],prompt)
                if manifest['processed_windows']!=job['planned_windows']:
                    raise ValueError('Cached window count differs from current planning')
                for name,expected_hash in job.get('artifact_sha256',{}).items():
                    if digest(directory/name)!=expected_hash:
                        raise ValueError('Planned cached artifact changed')
                x0,y0,x1,y1=manifest['crop_xyxy'];window=Window(x0,y0,x1-x0,y1-y0)
                raw=roi.read(1,window=window)
                mask=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=window)>0)
                target=gt.read(1,window=window)
                if np.any(target<0): raise ValueError('Negative GT labels')
                if int(mask.sum())!=full_roi: raise ValueError('Run crop omits ROI pixels')
                for name in ('instance','count'):
                    with rasterio.open(directory/f'{name}.tif') as src:
                        if src.shape!=mask.shape or not src.transform.almost_equals(image.window_transform(window)):
                            raise ValueError('Run output grid mismatch')
                        data=src.read(1)
                    if name=='instance': pred=data
                    elif np.any(mask&(data==0)): raise ValueError('ROI contains uncovered pixels')
                baseline=scores(roi_counts(base.read(1,window=window),target,mask,parameters['boundary_tolerance']))
                if baseline_scores is not None and baseline!=baseline_scores:
                    raise ValueError('Baseline metrics differ across run crops')
                baseline_scores=baseline
                metrics=scores(roi_counts(pred,target,mask,parameters['boundary_tolerance']))
                row=dict(tile_size=job['tile_size'],stride=job['stride'],overlap_fraction=job['overlap_fraction'],
                         windows=manifest['processed_windows'],seconds=manifest['seconds'],
                         window_seconds=sum(r['seconds'] for r in records),run_dir=str(directory),**metrics)
                rows.append(row)
                job['artifact_sha256']={name:digest(directory/name) for name in ('manifest.json','windows.jsonl','instance.tif','count.tif')}
        if any(digest(parameters[k])!=hashes[k] for k in PATH_KEYS) or digest(gt_path)!=hashes['gt']:
            raise ValueError('Sweep inputs changed')
        with (output_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        charts=[]
        for key,title in [('mask_iou','Foreground IoU'),('boundary_f1','Boundary F1'),('instance_f1','Instance F1'),('pq','PQ'),('windows','Processed windows'),('seconds','Run seconds (loading + IO included)')]:
            filename=key+'.svg'
            (output_dir/filename).write_text(curve(rows,key,title,baseline_scores.get(key)),encoding='utf-8')
            charts.append(f'<img src="{filename}" style="width:660px;max-width:100%">')
        page='<!doctype html><meta charset="utf-8"><title>Dense stride ablation</title><h1>Dense stride ablation</h1><p>'+html.escape(state['limitation'])+'</p><p>'+html.escape(state['legacy_reuse'])+'</p><p>Axes auto-scaled; absolute metric units (0 to 1). Hover over points for stride and value. No significance or monotonicity claim.</p>'+''.join(charts)
        (output_dir/'index.html').write_text(page,encoding='utf-8')
        state.update(status='complete',baseline=baseline_scores,rows=rows)
        state.pop('error',None);save()
        lines=['Dense stride ablation; fixed center fusion; development ROI only.',
               'BsiNet baseline: '+', '.join(f'{k}={baseline_scores[k]:.6f}' for k in ('mask_iou','boundary_f1','instance_f1','pq'))]
        for row in rows:
            lines.append(f"t{row['tile_size']}_s{row['stride']}: windows={row['windows']}, seconds={row['seconds']:.2f}, IoU={row['mask_iou']:.6f}, boundary_F1={row['boundary_f1']:.6f}, instance_F1={row['instance_f1']:.6f}, PQ={row['pq']:.6f}")
        (output_dir/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(lines),flush=True)
        print(f'Complete: {output_dir}/comparison.csv and index.html',flush=True)
        return state
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-run',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reuse-root',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    sweep(args.template_run,args.gt,args.output_dir,args.plan_only,args.reuse_root)


if __name__=='__main__':
    main()
