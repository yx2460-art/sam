"""Compare four cached fusion rules across every completed dense-stride configuration."""
import argparse
import csv
import html
import json
import shutil
from pathlib import Path

from .refuse_tiles import METHODS, refusion
from .run_tiles import digest
from .sweep_strides import PATH_KEYS, read_json, validate_run


def plot(rows,tile,metric,baseline):
    selected=[r for r in rows if r['tile_size']==tile]
    values=[r[metric] for r in selected]+[baseline]
    low,high=min(values),max(values)
    padding=max((high-low)*.12,.002)
    low,high=low-padding,high+padding
    def xy(row):
        return 75+row['overlap_fraction']/.75*560,235-(row[metric]-low)/(high-low)*185
    chunks=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 310">',
            '<rect width="700" height="310" fill="white"/>',
            f'<text x="75" y="20">{tile}px / {metric}</text>',
            '<path d="M75 50V235H635" stroke="black" fill="none"/>']
    for i in range(5):
        value=low+(high-low)*i/4
        chunks.append(f'<text x="4" y="{235-i*185/4:.1f}" font-size="11">{value:.4f}</text>')
    for value in (0,.125,.25,.375,.5,.625,.75):
        chunks.append(f'<text x="{75+value/.75*560-10:.1f}" y="254" font-size="11">{value*100:g}%</text>')
    y=235-(baseline-low)/(high-low)*185
    chunks.append(f'<path d="M75 {y:.2f}H635" stroke="#555" stroke-dasharray="5 4"/><text x="78" y="{y-4:.2f}" font-size="10">BsiNet</text>')
    for index,(method,color) in enumerate(zip(METHODS,('#1764b0','#20813d','#bd3030','#9146b0'))):
        data=sorted([r for r in selected if r['method']==method],key=lambda r:r['overlap_fraction'])
        points=' '.join(f'{xy(r)[0]:.2f},{xy(r)[1]:.2f}' for r in data)
        chunks.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        chunks.append(f'<text x="{75+index*145}" y="295" font-size="12" fill="{color}">{method}</text>')
        for row in data:
            x,y=xy(row)
            chunks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>stride={row["stride"]}; {row[metric]:.6f}</title></circle>')
    chunks.append('<text x="280" y="274" font-size="12">Overlap per axis</text></svg>')
    return ''.join(chunks)


def output_hashes(directory):
    names=['comparison.json','comparison.csv']
    names += [f'{method}/{name}' for method in METHODS
              for name in ('manifest.json','summary.json','instance.tif','count.tif')]
    return {name:digest(directory/name) for name in names}


def sweep(sweep_dir,output_dir,plan_only=False):
    sweep_dir,output_dir=Path(sweep_dir).resolve(),Path(output_dir).resolve()
    source_path=sweep_dir/'sweep.json'
    source=read_json(source_path)
    if source.get('status')!='complete':
        raise ValueError('Dense stride sweep must be complete')
    signature=source['signature']
    parameters=signature['parameters']
    gt_path=Path(signature['gt'])
    for key in PATH_KEYS:
        if digest(parameters[key])!=signature['hashes'][key]:
            raise ValueError(f'Changed dense sweep input: {key}')
    if digest(gt_path)!=signature['hashes']['gt']:
        raise ValueError('GT changed since dense evaluation')
    if output_dir==sweep_dir or sweep_dir in output_dir.parents:
        raise ValueError('Fusion outputs must be outside the dense sweep')
    baseline=source['baseline']
    definitions=[(job['tile_size'],job['stride']) for job in source['jobs']]
    if len(set(definitions))!=len(definitions) or set(definitions)!={tuple(c) for c in signature['configurations']}:
        raise ValueError('Dense sweep configurations are missing/duplicated')
    metric_rows={(r['tile_size'],r['stride']):r for r in source['rows']}
    if len(source['rows'])!=len(definitions) or set(metric_rows)!=set(definitions):
        raise ValueError('Dense sweep metric rows are incomplete')
    job_signature=dict(source_sweep=str(source_path),source_sweep_sha256=digest(source_path),
                       input_hashes=signature['hashes'],methods=list(METHODS))
    state_path=output_dir/'fusion_sweep.json'
    if output_dir.exists():
        if not state_path.is_file() or read_json(state_path)['signature']!=job_signature:
            raise ValueError('Existing output is not a matching fusion sweep')
        state=read_json(state_path)
    else:
        state=dict(status='planning',signature=job_signature,jobs=[],
                   sam_inference_performed=False,
                   notes='Fixed BsiNet IDs and binary window votes; all settings reported without GT-based selection. Developer ROI, training independence unverified.',
                   cache_scope='Cache hashes are snapshots taken now, not historical inference seals; refusion validates invariants and reproduces original center output.')
    previous={(j['tile_size'],j['stride']):j for j in state['jobs']}
    jobs=[]
    bytes_needed=0
    for job in source['jobs']:
        tile,stride=job['tile_size'],job['stride']
        directory=Path(job['run_dir'])
        if output_dir==directory or directory in output_dir.parents:
            raise ValueError('Output is inside a source window run')
        validate_run(directory,parameters,signature['hashes'],tile,stride,signature['prompt'])
        for name,expected in job['artifact_sha256'].items():
            if digest(directory/name)!=expected:
                raise ValueError(f'Changed dense run artifact: {directory/name}')
        manifest=read_json(directory/'manifest.json')
        x0,y0,x1,y1=manifest['crop_xyxy']
        pixels=(x1-x0)*(y1-y0)
        cache_hashes={}
        for key in ('count','votes','center'):
            path=directory/'work'/f'{key}.bin'
            if not path.is_file() or path.stat().st_size!=pixels*(1 if key=='center' else 4):
                raise ValueError(f'Missing/wrong-sized cache: {path}. No inference was started.')
            cache_hashes[key]=digest(path)
        old=previous.get((tile,stride))
        if old and old['cache_sha256']!=cache_hashes:
            raise ValueError('Source cache changed after fusion planning')
        destination=output_dir/f't{tile}_s{stride}'
        complete=False
        if destination.exists():
            report_path=destination/'comparison.json'
            if not report_path.is_file() or read_json(report_path).get('status')!='complete':
                raise ValueError(f'Partial fusion cannot be overwritten: {destination}. Preserve/relocate that partial configuration before retrying, or use a new output directory.')
            report=read_json(report_path)
            if (report['source_run']!=str(directory.resolve()) or report['methods']!=list(METHODS) or
                    report.get('gt_sha256')!=signature['hashes']['gt'] or report['cache_sha256']!=cache_hashes or
                    report['source_manifest_sha256']!=digest(directory/'manifest.json')):
                raise ValueError('Completed fusion provenance does not match dense run')
            current_hashes=output_hashes(destination)
            if old and old.get('output_sha256') and old['output_sha256']!=current_hashes:
                raise ValueError('Previously summarized fusion outputs changed')
            complete=True
        else:
            # Uncompressed labels + count raster, four methods; sidecars/filesystem extra.
            bytes_needed+=pixels*8*len(METHODS)
        item=dict(tile_size=tile,stride=stride,run_dir=str(directory),output_dir=str(destination),
                  overlap_fraction=job['overlap_fraction'],cache_sha256=cache_hashes,
                  status='complete' if complete else 'planned')
        if complete: item['output_sha256']=current_hashes
        jobs.append(item)
    output_dir.mkdir(parents=True,exist_ok=True)
    state.update(status='planned',jobs=jobs)
    def save():
        state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    pending=sum(j['status']!='complete' for j in jobs)
    print(f'{len(jobs)} configurations x {len(METHODS)} methods; {pending} configurations need refusion; no SAM inference.',flush=True)
    print(f'New label/count rasters uncompressed estimate={bytes_needed/2**30:.2f} GiB; free disk={shutil.disk_usage(output_dir).free/2**30:.2f} GiB. Compression varies; sidecars extra.',flush=True)
    if plan_only:
        return state
    try:
        state['status']='running';save()
        rows=[]
        metrics=tuple(baseline)
        for job in jobs:
            destination=Path(job['output_dir'])
            directory=Path(job['run_dir'])
            if job['status']!='complete':
                print(f"Re-fusing t{job['tile_size']}_s{job['stride']}",flush=True)
                refusion(directory,destination,gt_path,methods=METHODS)
                job.update(status='complete',output_sha256=output_hashes(destination));save()
            comparison=read_json(destination/'comparison.json')
            for method in METHODS:
                entry=comparison['results'][method]
                summary=entry['summary']
                if any(summary['baseline'][k]!=baseline[k] for k in metrics):
                    raise ValueError('Baseline differs across fusion configurations')
                result=summary['sam']
                center=metric_rows[(job['tile_size'],job['stride'])]
                if method=='center' and any(result[k]!=center[k] for k in metrics):
                    raise ValueError('Reproduced center metrics differ from original dense sweep')
                if job['stride']==job['tile_size'] and any(result[k]!=center[k] for k in metrics):
                    raise ValueError('Fusion rules differ at no overlap; cache/grid consistency requires review')
                row=dict(tile_size=job['tile_size'],stride=job['stride'],overlap_fraction=job['overlap_fraction'],
                    method=method,windows=center['windows'],source_inference_seconds=center['seconds'],
                    fusion_seconds=entry['fusion_seconds'],**{k:result[k] for k in metrics})
                for key in ('mask_iou','boundary_f1','instance_f1','pq'):
                    row['delta_'+key+'_vs_bsinet']=result[key]-baseline[key]
                    row['delta_'+key+'_vs_center']=result[key]-center[key]
                rows.append(row)
        if digest(source_path)!=job_signature['source_sweep_sha256']:
            raise ValueError('Source sweep changed during refusion')
        for key in PATH_KEYS:
            if digest(parameters[key])!=signature['hashes'][key]:
                raise ValueError(f'Sweep input changed during refusion: {key}')
        if digest(gt_path)!=signature['hashes']['gt']:
            raise ValueError('GT changed during refusion')
        for job in jobs:
            if any(digest(Path(job['run_dir'])/'work'/f'{k}.bin')!=v for k,v in job['cache_sha256'].items()):
                raise ValueError('Cache changed during fusion sweep')
            if output_hashes(Path(job['output_dir']))!=job['output_sha256']:
                raise ValueError('Fusion output changed during sweep')
        with (output_dir/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        panels=[]
        for tile in sorted({j['tile_size'] for j in jobs}):
            for key in ('mask_iou','boundary_f1','instance_f1','pq'):
                name=f't{tile}_{key}.svg'
                (output_dir/name).write_text(plot(rows,tile,key,baseline[key]),encoding='utf-8')
                panels.append(f'<img src="{name}" style="width:700px;max-width:100%">')
        page='<!doctype html><meta charset="utf-8"><title>Stride and fusion ablation</title><h1>Stride and fusion ablation</h1><p>'+html.escape(state['notes'])+'</p><p>Metric units: 0-1. Axes auto-scaled. Dashed line: BsiNet. All four rules share the same cached window predictions within each configuration. No automatic selection of a best rule or stride.</p>'+''.join(panels)
        (output_dir/'index.html').write_text(page,encoding='utf-8')
        lines=['Stride x fusion comparison; no SAM inference; development ROI only.',
               'BsiNet: '+', '.join(f'{k}={baseline[k]:.6f}' for k in ('mask_iou','boundary_f1','instance_f1','pq'))]
        for row in rows:
            lines.append(f"t{row['tile_size']}_s{row['stride']} / {row['method']}: IoU={row['mask_iou']:.6f}, boundary_F1={row['boundary_f1']:.6f}, instance_F1={row['instance_f1']:.6f}, PQ={row['pq']:.6f}, fusion_seconds={row['fusion_seconds']:.3f}")
        text='\n'.join(lines)+'\n'
        (output_dir/'summary.txt').write_text(text,encoding='utf-8')
        state.update(status='complete',baseline=baseline,rows=rows)
        state.pop('error',None);save()
        print(text,flush=True)
        print(f'Complete: {output_dir}/comparison.csv and index.html',flush=True)
        return state
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    sweep(args.sweep_dir,args.output_dir,args.plan_only)


if __name__=='__main__':
    main()
