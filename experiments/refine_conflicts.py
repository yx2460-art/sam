"""Union baseline vs conflict-targeted and area-matched random local SAM refinement."""
import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from scipy import ndimage
from rasterio.windows import Window

from .conflict_regions import plan_regions,random_controls,region_mask
from .metrics import scores
from .raster import save_labels
from .run_tiles import aligned,digest
from .tiles import source_ids


def save_mask(path,mask,profile):
    cfg=profile.copy()
    cfg.update(driver='GTiff',count=1,dtype='uint8',nodata=None,compress='deflate')
    for key in ('photometric','interleave'):
        cfg.pop(key,None)
    with rasterio.open(path,'w',**cfg) as dst:
        dst.write(mask.astype(np.uint8),1)


def run(args,refine=None,predictor=None):
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new directory: {args.output_dir}')
    if args.context<0 or args.seed<0:
        raise ValueError('context and seed must be nonnegative')
    union_dir=args.union_dir.resolve()
    manifest_path=union_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or manifest.get('fusion')!='union':
        raise ValueError('Expected a completed cached union-fusion run')
    source_dir=Path(manifest['source_run'])
    source_path=source_dir/'manifest.json'
    if digest(source_path)!=manifest['source_manifest_sha256']:
        raise ValueError('Original window-run manifest changed')
    src_manifest=json.loads(source_path.read_text(encoding='utf-8'))
    params=manifest['parameters']
    if digest(args.checkpoint)!=manifest['hashes']['checkpoint']:
        raise ValueError('Use the same SAM checkpoint as the first-stage windows')
    for key in ('prediction','eval_mask'):
        if digest(params[key])!=manifest['hashes'][key]:
            raise ValueError(f'Changed source input: {key}')
    if args.output_dir.resolve()==union_dir or union_dir in args.output_dir.resolve().parents:
        raise ValueError('Output must be outside union source run')
    started=time.perf_counter()
    with ExitStack() as stack:
        image=stack.enter_context(rasterio.open(params['image']))
        base=stack.enter_context(rasterio.open(params['prediction']))
        roi=stack.enter_context(rasterio.open(params['eval_mask']))
        coverage=stack.enter_context(rasterio.open(params['coverage']))
        original_union=stack.enter_context(rasterio.open(union_dir/'instance.tif'))
        allow=params.get('allow_crs_mismatch',False)
        for src in (base,roi,coverage):
            aligned(image,src,allow)
        x0,y0,x1,y1=manifest['crop_xyxy']
        window=Window(x0,y0,x1-x0,y1-y0)
        shape=(y1-y0,x1-x0)
        expected=SimpleNamespace(width=shape[1],height=shape[0],crs=image.crs,
                                 transform=image.window_transform(window))
        aligned(expected,original_union,allow)
        profile=original_union.profile.copy()
        counts={}
        for key in ('count','votes'):
            path=source_dir/'work'/f'{key}.bin'
            if not path.is_file() or path.stat().st_size!=int(np.prod(shape))*4:
                raise ValueError(f'Missing/wrong-sized cache: {path}')
            if digest(path)!=manifest['cache_sha256'][key]:
                raise ValueError(f'Cache changed since union fusion: {key}')
            counts[key]=np.memmap(path,mode='r',dtype='uint32',shape=shape)
        c,v=counts['count'],counts['votes']
        if np.any(v>c):
            raise ValueError('Invalid vote counts')
        initial=original_union.read(1)
        baseline=base.read(1,window=window)
        expected_union=np.where(v>0,baseline,0)
        if not np.array_equal(initial,expected_union):
            raise ValueError('Union raster no longer matches the cached window votes')
        raw_roi=roi.read(1,window=window)
        selected=(raw_roi>0)&np.isfinite(raw_roi)&(roi.read_masks(1,window=window)>0)
        if np.any(selected&(c==0)):
            raise ValueError('ROI contains unobserved pixels')
        bands=params.get('rgb_bands',[1,2,3])
        valid=np.all(image.read_masks(bands,window=window)>0,axis=0)
        valid &= coverage.read(1,window=window)>0
        domain=selected&valid
        if not np.array_equal(domain,selected):
            raise ValueError('Scoring mask includes invalid pixels; do not silently drop them')
        conflict=(v>0)&(v<c)&domain
        regions,total_candidates=plan_regions(conflict,domain,args.core_size,args.write_buffer,
                                              args.min_conflict,args.max_regions)
        if not regions:
            raise ValueError('No qualifying conflict regions; no inference performed')
        conflict_write=region_mask(regions,shape)
        random_regions=random_controls(regions,domain,conflict_write|conflict,
                                       baseline>0,args.seed)
        random_write=region_mask(random_regions,shape)
        if int(random_write.sum())!=int(conflict_write.sum()):
            raise ValueError('Write budgets are not equal')
        args.output_dir.mkdir(parents=True)
        plans={'conflict':regions,'random':random_regions}
        report=dict(status='planned',ground_truth_used_for_planning_or_inference=False,
                    source_union=str(union_dir),source_union_sha256=digest(union_dir/'instance.tif'),
                    source_manifest_sha256=digest(manifest_path),
                    checkpoint_sha256=manifest['hashes']['checkpoint'],
                    settings={key:str(value) if isinstance(value,Path) else value
                              for key,value in vars(args).items()},
                    total_conflict_pixels=int(conflict.sum()),qualifying_regions=total_candidates,
                    processed_regions=len(regions),conflict_pixels_targeted=int((conflict_write&conflict).sum()),
                    write_pixels_per_method=int(conflict_write.sum()),
                    matching='same translated write-mask shapes and context dimensions; fixed seed; '
                             'predicted foreground fraction within 20 percentage points; '
                             'actual decoder calls/time are reported, not guaranteed equal',
                    scope='fixed BsiNet IDs, no new instance discovery or splitting; developer ROI only',
                    results={})
        plan_json={}
        for method,items in plans.items():
            plan_json[method]=[dict(x_full=x0+r['x'],y_full=y0+r['y'],
                                    width=r['mask'].shape[1],height=r['mask'].shape[0],
                                    write_pixels=int(r['mask'].sum())) for r in items]
        report['plans']=plan_json
        (args.output_dir/'plan.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        save_mask(args.output_dir/'conflict.tif',conflict,profile)
        save_mask(args.output_dir/'conflict_write.tif',conflict_write,profile)
        save_mask(args.output_dir/'random_write.tif',random_write,profile)
        if args.plan_only:
            print(f'Plan only: {len(regions)} regions per method; {int(conflict_write.sum())} write pixels each')
            return report
        if refine is None:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'segment-anything'))
            import torch
            from segment_anything import SamPredictor,sam_model_registry
            from .refine import refine_image
            if args.device.startswith('cuda') and not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            model=sam_model_registry[params.get('model_type','vit_b')](checkpoint=str(args.checkpoint))
            model.to(device=args.device).eval()
            predictor=SamPredictor(model)
            def refine(rgb,original,predictor,options):
                with torch.inference_mode():
                    return refine_image(rgb,original,predictor,options)
        options=SimpleNamespace(**src_manifest['fixed_prompt_parameters'])
        if options.mode!='interior' or options.support_pad!=0 or options.residual_policy!='replace':
            raise ValueError('This second-stage experiment requires the fixed interior/shrink protocol')
        outputs={}
        for method,items in plans.items():
            directory=args.output_dir/method
            directory.mkdir()
            result=initial.copy()
            logs=[]
            method_start=time.perf_counter()
            calls=0
            for i,region in enumerate(items):
                x,y=region['x'],region['y']
                h,w=region['mask'].shape
                # Fixed context dimensions; boundless reads keep edge budgets matched.
                context=Window(x0+x-args.context,y0+y-args.context,
                               w+2*args.context,h+2*args.context)
                rgb=image.read(bands,window=context,boundless=True,fill_value=0)
                ok=np.all(image.read_masks(bands,window=context,boundless=True)>0,axis=0)
                ok &= coverage.read(1,window=context,boundless=True,fill_value=0)>0
                ok &= np.all(np.isfinite(rgb),axis=0)
                original=base.read(1,window=context,boundless=True,fill_value=0)
                original=np.where(ok,original,0).astype(np.uint32)
                if params.get('rgb_range'):
                    low,high=params['rgb_range']
                    rgb=np.clip((rgb.astype(np.float32)-low)*255/(high-low),0,255).astype(np.uint8)
                rgb=np.moveaxis(rgb,0,-1).copy()
                rgb[~ok]=0
                tick=time.perf_counter()
                local,details=refine(rgb,original,predictor,options)
                calls+=sum(r.get('prompt_count',0) for r in details)
                candidate=source_ids(local,original)
                accepted_components=np.zeros(original.shape,dtype=bool)
                component_cache={}
                for record in details:
                    if record.get('accepted_count',0)<=0:
                        continue
                    lab=record['source_id']
                    if lab not in component_cache:
                        component_cache[lab]=ndimage.label(original==lab,structure=np.ones((3,3)))[0]
                    accepted_components |= component_cache[lab]==record['component']
                core=np.s_[args.context:args.context+h,args.context:args.context+w]
                # A fallback to the old BsiNet mask is NOT a successful refinement:
                # retain union pixels for any source ID with no accepted candidate.
                update=region['mask']&accepted_components[core]&ok[core]
                result[y:y+h,x:x+w][update]=candidate[core][update]
                save_labels(directory/f'patch_{i:03d}.tif',candidate,
                            dict(profile,width=candidate.shape[1],height=candidate.shape[0],
                                 transform=image.window_transform(context)))
                logs.append(dict(index=i,x_full=int(context.col_off),y_full=int(context.row_off),
                                 context_width=int(context.width),context_height=int(context.height),
                                 requested_write_pixels=int(region['mask'].sum()),
                                 eligible_write_pixels=int(update.sum()),seconds=time.perf_counter()-tick,
                                 objects=details))
                print(f'{method} region {i+1}/{len(items)}: eligible write {int(update.sum())}',flush=True)
            method_write=conflict_write if method=='conflict' else random_write
            if np.any((result!=initial)&~method_write):
                raise ValueError('Unexpected write outside the planned region')
            save_labels(directory/'instance.tif',result,profile)
            (directory/'patches.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
            outputs[method]=result
            report['results'][method]=dict(seconds=time.perf_counter()-method_start,prompt_calls=calls,
                                           changed_pixels=int((result!=initial).sum()),
                                           added_pixels=int(((result>0)&(initial==0)).sum()),
                                           removed_pixels=int(((result==0)&(initial>0)).sum()),
                                           eligible_write_pixels=sum(r['eligible_write_pixels'] for r in logs))
        report['status']='inference_complete'
        report['seconds']=time.perf_counter()-started
        revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,
                                cwd=Path(__file__).resolve().parents[1],check=False)
        report['code_revision']=revision.stdout.strip()
        # No truth is opened until ALL planning and both inference arms have finished.
        if args.gt:
            from .evaluate_tiles import roi_counts
            gt=stack.enter_context(rasterio.open(args.gt))
            aligned(image,gt,allow)
            if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
                raise ValueError('GT must be integer instance labels with NoData=None/0')
            target=gt.read(1,window=window)
            if np.any(target<0):
                raise ValueError('Negative GT IDs')
            report['gt_sha256']=digest(args.gt)
            rows=[]
            for name,data in [('bsinet',baseline),('union',initial),*outputs.items()]:
                counts=roi_counts(data,target,domain,params.get('boundary_tolerance',2))
                entry=dict(scores(counts),counts=counts)
                report.setdefault('evaluation',{})[name]=entry
                rows.append(dict(method=name,seconds=report['results'].get(name,{}).get('seconds',0),
                                 **scores(counts)))
                # Fixed ORIGINAL conflict domain, not a method-dependent post-hoc ROI.
                local_counts=roi_counts(data,target,conflict,params.get('boundary_tolerance',2))
                report.setdefault('conflict_evaluation',{})[name]=dict(scores(local_counts),counts=local_counts)
                consistent=domain&~conflict
                if consistent.any():
                    local_counts=roi_counts(data,target,consistent,params.get('boundary_tolerance',2))
                    report.setdefault('consistent_evaluation',{})[name]=dict(scores(local_counts),counts=local_counts)
            with (args.output_dir/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
        report['status']='complete'
        report['evaluation_warning']='ROI-clipped developer metrics; independence unverified. Narrow '
        report['evaluation_warning']+='conflict masks may leave zero boundary-interior pixels; inspect counts.'
        if digest(union_dir/'instance.tif')!=report['source_union_sha256']:
            raise ValueError('Source union changed during experiment')
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'Complete; saved {args.output_dir}',flush=True)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--union-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gt',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--core-size',type=int,default=256)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--write-buffer',type=int,default=8)
    p.add_argument('--min-conflict',type=int,default=32)
    p.add_argument('--max-regions',type=int,default=16)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--plan-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
