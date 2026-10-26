"""BsiNet-guided full-image window/stride experiments, ground-truth-free inference."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import Window

from .check_full import grid_errors
from .raster import save_labels
from .tiles import center_weights, fuse_result, mask_bounds, source_ids, windows


def aligned(reference, source, allow):
    errors = grid_errors(reference, source)
    if errors and not (allow and errors == ['CRS differs']):
        raise ValueError(f'{source.name}: {errors}')
    if source.count != 1:
        raise ValueError(f'Expected one band: {source.name}')
    return errors


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def run(args, refine=None, predictor=None):
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    if args.context < 0 or args.max_windows < 0:
        raise ValueError('context and max-windows must be nonnegative')
    if args.rgb_range and args.rgb_range[1] <= args.rgb_range[0]:
        raise ValueError('Invalid RGB value range')
    started = time.perf_counter()
    with ExitStack() as stack:
        image = stack.enter_context(rasterio.open(args.image))
        base = stack.enter_context(rasterio.open(args.prediction))
        coverage = stack.enter_context(rasterio.open(args.coverage))
        roi = stack.enter_context(rasterio.open(args.eval_mask))
        overrides = {}
        for key, src in [('prediction', base), ('coverage', coverage), ('eval_mask', roi)]:
            overrides[key] = aligned(image, src, args.allow_crs_mismatch)
        if not np.issubdtype(np.dtype(base.dtypes[0]), np.integer) or base.nodata not in (None, 0):
            raise ValueError('Prediction must contain integer IDs with NoData=None/0')
        bands = args.rgb_bands
        if len(set(bands)) != 3 or min(bands) < 1 or max(bands) > image.count:
            raise ValueError('Specify three distinct existing RGB bands')
        if any(image.dtypes[b-1] != 'uint8' for b in bands) and args.rgb_range is None:
            raise ValueError('Non-UInt8 RGB requires --rgb-range LOW HIGH')
        selected_bounds = mask_bounds(roi, args.context)
        planned = list(windows(image.width, image.height, args.tile_size, args.stride, selected_bounds))
        if not planned:
            raise ValueError('No windows')
        if args.max_windows:
            # A smoke run must actually exercise a foreground window near the ROI,
            # not spend its only iteration on the surrounding invalid image border.
            used=[]
            for win in planned:
                if np.any(roi.read(1,window=win)>0) and np.any(base.read(1,window=win)>0):
                    used.append(win)
                    if len(used)>=args.max_windows:
                        break
            if not used:
                raise ValueError('No smoke window intersects both ROI and predicted foreground')
        else:
            used=planned
        # Include the complete selected windows, not just the scoring mask bbox.
        x0 = min(int(w.col_off) for w in used)
        y0 = min(int(w.row_off) for w in used)
        x1 = max(int(w.col_off+w.width) for w in used)
        y1 = max(int(w.row_off+w.height) for w in used)
        crop = Window(x0, y0, x1-x0, y1-y0)
        args.output_dir.mkdir(parents=True)
        work = args.output_dir / 'work'
        work.mkdir()
        shape = (int(crop.height), int(crop.width))
        def disk(name, dtype):
            a = np.memmap(work / f'{name}.bin', dtype=dtype, mode='w+', shape=shape)
            a[:] = 0
            return a
        count, votes = disk('count', 'uint32'), disk('votes', 'uint32')
        weight, center = disk('weight', 'float32'), disk('center', 'uint8')
        parameters = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                  text=True, capture_output=True, check=False)
        manifest = dict(status='running', parameters=parameters, grid_origin=[0,0],
                        crop_xyxy=[x0,y0,x1,y1], planned_windows=len(planned),
                        requested_windows=len(used), crs_overrides=overrides,
                        inference_ground_truth_used=False, eval_mask_used='window extent only, not prompts',
                        instance_identity='fixed BsiNet source ID; no new parcel split',
                        fusion=args.fusion, code_revision=revision.stdout.strip(),
                        hashes={k:digest(v) for k,v in [('eval_mask',args.eval_mask),
                                                      ('prediction',args.prediction),('checkpoint',args.checkpoint)]},
                        versions={'numpy':np.__version__, 'rasterio':rasterio.__version__})
        def manifest_write():
            (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        manifest_write()
        if refine is None:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'segment-anything'))
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            from .refine import refine_image
            if args.device.startswith('cuda') and not torch.cuda.is_available():
                raise ValueError('CUDA unavailable; specify --device cpu if intended')
            model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
            model.to(device=args.device).eval()
            predictor = SamPredictor(model)
            def refine(rgb, original, predictor, options):
                with torch.inference_mode():
                    return refine_image(rgb,original,predictor,options)
            manifest['versions']['torch'] = torch.__version__
        options = SimpleNamespace(mode='interior', residual_policy='replace', box_pad=2,
                                  support_pad=0, min_area=30, min_score=0.8, min_stability=0.9,
                                  stability_offset=1, min_support=0.8, min_coverage=0.5,max_overlap=0.05)
        manifest['fixed_prompt_parameters'] = vars(options)
        tile_dir = args.output_dir/'tiles'
        tile_dir.mkdir()
        records = (args.output_dir/'windows.jsonl').open('w',encoding='utf-8')
        stack.callback(records.close)
        for index, window in enumerate(used):
            tick = time.perf_counter()
            rgb = image.read(bands,window=window)
            valid = np.all(image.read_masks(bands,window=window)>0,axis=0)
            valid &= np.all(np.isfinite(rgb),axis=0)
            cov = coverage.read(1,window=window)
            if not np.all(np.isin(cov,[0,1,255])):
                raise ValueError('Coverage must be binary 0/1 or 0/255')
            valid &= cov>0
            original = base.read(1,window=window)
            if np.any(original<0) or np.any(original>np.iinfo(np.uint32).max):
                raise ValueError('Invalid source IDs')
            original = np.where(valid,original,0).astype(np.uint32)
            if args.rgb_range:
                low,high=args.rgb_range
                rgb=np.clip((rgb.astype(np.float32)-low)*255/(high-low),0,255).astype(np.uint8)
            rgb=np.moveaxis(rgb,0,-1).copy()
            rgb[~valid]=0
            logs=[]
            if original.any():
                local,logs=refine(rgb,original,predictor,options)
                result=source_ids(local,original)
            else:
                result=np.zeros(original.shape,dtype=np.uint32)
            y,x=int(window.row_off)-y0,int(window.col_off)-x0
            h,w=original.shape
            target=np.s_[y:y+h,x:x+w]
            count[target]+=valid.astype(np.uint32)
            votes[target]+=((result>0)&valid).astype(np.uint32)
            candidate_weight=center_weights(h,w)
            better=valid & (candidate_weight>weight[target])
            weight[target][better]=candidate_weight[better]
            center[target][better]=(result[better]>0).astype(np.uint8)
            profile=image.profile.copy()
            profile.update(width=w,height=h,transform=image.window_transform(window))
            tile_path=tile_dir/f'{index:05d}.tif'
            save_labels(tile_path,result,profile)
            row=dict(index=index,x=int(window.col_off),y=int(window.row_off),width=w,height=h,
                     valid_pixels=int(valid.sum()),seconds=time.perf_counter()-tick,
                     tile=str(tile_path.relative_to(args.output_dir)),objects=logs)
            records.write(json.dumps(row)+'\n'); records.flush()
            print(f'Window {index+1}/{len(used)}: x={row["x"]}, y={row["y"]}, {row["seconds"]:.2f}s',flush=True)
        out_profile=image.profile.copy()
        out_profile.update(driver='GTiff',width=shape[1],height=shape[0],count=1,dtype='uint32',
                           transform=image.window_transform(crop),nodata=None,compress='deflate')
        for key in ('photometric','interleave'):
            out_profile.pop(key,None)
        with rasterio.open(args.output_dir/'instance.tif','w',**out_profile) as dst, \
             rasterio.open(args.output_dir/'count.tif','w',**out_profile) as counts, \
             rasterio.open(args.output_dir/'conflict.tif','w',**dict(out_profile,dtype='uint8')) as conflicts:
            for y in range(0,shape[0],512):
                for x in range(0,shape[1],512):
                    win=Window(x,y,min(512,shape[1]-x),min(512,shape[0]-y))
                    sl=np.s_[y:y+int(win.height),x:x+int(win.width)]
                    original=base.read(1,window=Window(x+x0,y+y0,win.width,win.height))
                    fused=fuse_result(original,votes[sl],count[sl],center[sl],args.fusion)
                    dst.write(fused,1,window=win)
                    counts.write(count[sl],1,window=win)
                    conflict=(votes[sl]>0)&(votes[sl]<count[sl])
                    conflicts.write(conflict.astype(np.uint8),1,window=win)
        for a in (count,votes,weight,center):
            a.flush()
        # Keep scratch maps for reproducibility/re-fusion; document disk cost.
        manifest.update(status='smoke_only' if args.max_windows else 'complete',
                        processed_windows=len(used),seconds=time.perf_counter()-started,
                        evaluation_ready=False, scratch_bytes=sum(p.stat().st_size for p in work.iterdir()))
        manifest_write()
        print(f"Run {manifest['status']}; saved {args.output_dir}",flush=True)
        if args.gt and not args.max_windows:
            from .evaluate_tiles import evaluate_run
            evaluate_run(args.output_dir,args.gt,args.eval_mask,args.allow_crs_mismatch,args.boundary_tolerance)
        elif args.gt:
            print('Smoke run: partial coverage, evaluation deliberately skipped.')
        return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('image','prediction','coverage','eval-mask','checkpoint','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tile-size',type=int,required=True)
    p.add_argument('--stride',type=int,required=True)
    p.add_argument('--context',type=int,default=1024,help='Fixed ROI bbox expansion for all scales')
    p.add_argument('--max-windows',type=int,default=0,help='Positive=smoke only, no metrics')
    p.add_argument('--fusion',choices=['center','union','intersection','majority'],default='center')
    p.add_argument('--model-type',choices=['vit_b','vit_l','vit_h'],default='vit_b')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rgb-bands',type=int,nargs=3,default=[1,2,3])
    p.add_argument('--rgb-range',type=float,nargs=2)
    p.add_argument('--allow-crs-mismatch',action='store_true')
    p.add_argument('--gt',type=Path,help='Optional labels read only AFTER inference for ROI evaluation')
    p.add_argument('--boundary-tolerance',type=int,default=2)
    args=p.parse_args()
    if args.boundary_tolerance<0:
        p.error('boundary-tolerance must be nonnegative')
    run(args)


if __name__=='__main__':
    main()
