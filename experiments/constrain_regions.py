"""Replay cached region updates inside a fixed union-instance boundary band."""
import argparse
import csv
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .diagnose_regions import transitions
from .evaluate_tiles import roi_counts
from .metrics import label_boundary, scores
from .raster import save_labels
from .refine_conflicts import save_mask
from .run_tiles import aligned, digest


def boundary_band(labels, radius):
    if radius < 0:
        raise ValueError('Boundary radius must be nonnegative')
    # Detect boundaries before applying any ROI/write mask: no artificial ROI edges.
    return ndimage.maximum_filter(label_boundary(labels),size=2*radius+1,mode='constant') > 0


def constrain(initial, candidate, allowed):
    result=initial.copy()
    result[allowed]=candidate[allowed]
    return result


def run(run_dir, output_dir, gt_path=None, radius=2):
    run_dir,output_dir=Path(run_dir).resolve(),Path(output_dir).resolve()
    if radius < 0:
        raise ValueError('Boundary radius must be nonnegative')
    if output_dir.exists():
        raise FileExistsError(f'Use a new directory: {output_dir}')
    source_report=run_dir/'report.json'
    report=json.loads(source_report.read_text(encoding='utf-8'))
    if report.get('status')!='complete':
        raise ValueError('Expected completed two-arm region experiment')
    union_dir=Path(report['source_union'])
    manifest_path=union_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for protected in (run_dir,union_dir.resolve(),Path(manifest['source_run']).resolve()):
        if output_dir==protected or protected in output_dir.parents:
            raise ValueError('Output must be outside source experiment directories')
    if manifest.get('fusion')!='union' or manifest.get('status')!='complete':
        raise ValueError('Expected completed union fusion')
    if digest(manifest_path)!=report['source_manifest_sha256']:
        raise ValueError('Union manifest changed')
    if digest(union_dir/'instance.tif')!=report['source_union_sha256']:
        raise ValueError('Union raster changed')
    params=manifest['parameters']
    for key in ('prediction','eval_mask'):
        if digest(params[key])!=manifest['hashes'][key]:
            raise ValueError(f'Source changed: {key}')
    source_manifest=Path(manifest['source_run'])/'manifest.json'
    if digest(source_manifest)!=manifest['source_manifest_sha256']:
        raise ValueError('First-stage manifest changed')
    inputs=[source_report,manifest_path,source_manifest,union_dir/'instance.tif',
            Path(params['prediction']),Path(params['eval_mask'])]
    hashes={str(p):digest(p) for p in inputs}
    start=time.perf_counter()
    with ExitStack() as stack:
        initial_src=stack.enter_context(rasterio.open(union_dir/'instance.tif'))
        baseline_src=stack.enter_context(rasterio.open(params['prediction']))
        roi_src=stack.enter_context(rasterio.open(params['eval_mask']))
        allow=params.get('allow_crs_mismatch',False)
        aligned(baseline_src,roi_src,allow)
        x0,y0,x1,y1=manifest['crop_xyxy']
        window=Window(x0,y0,x1-x0,y1-y0)
        if initial_src.shape!=(y1-y0,x1-x0) or not initial_src.transform.almost_equals(baseline_src.window_transform(window)):
            raise ValueError('Union crop grid mismatch')
        if not allow and initial_src.crs!=baseline_src.crs:
            raise ValueError('Union crop CRS mismatch')
        initial=initial_src.read(1)
        baseline=baseline_src.read(1,window=window)
        raw=roi_src.read(1,window=window)
        domain=(raw>0)&np.isfinite(raw)&(roi_src.read_masks(1,window=window)>0)
        if not domain.any():
            raise ValueError('Empty ROI')
        if np.any((initial>0)&(initial!=baseline)):
            raise ValueError('Union contains unexpected source IDs')
        band=boundary_band(initial,radius)
        # Crop-edge boundary proximity cannot be inferred from pixels outside this cache.
        # Conservatively protect the outer radius+1 pixels, equally for both arms.
        safe=np.zeros(initial.shape,dtype=bool)
        margin=radius+1
        if min(initial.shape)>2*margin:
            safe[margin:-margin,margin:-margin]=True
        band &= safe & domain
        cached={}
        outputs={}
        masks={}
        for method in ('conflict','random'):
            arrays=[]
            for path in (run_dir/method/'instance.tif',run_dir/f'{method}_write.tif'):
                hashes[str(path)]=digest(path)
                src=stack.enter_context(rasterio.open(path))
                aligned(initial_src,src,allow)
                data=src.read(1)
                if not np.issubdtype(data.dtype,np.integer) or np.any(data<0):
                    raise ValueError(f'Invalid labels: {path}')
                arrays.append(data)
            candidate,write=arrays
            if np.any((write!=0)&(write!=1)):
                raise ValueError('Nonbinary write mask')
            write=write>0
            if int(write.sum())!=report['write_pixels_per_method'] or np.any(write&~domain):
                raise ValueError('Write mask budget/ROI changed')
            if np.any((candidate!=initial)&~write) or np.any((candidate>0)&(candidate!=baseline)):
                raise ValueError('Cached updates violate source ID/write constraints')
            actual=dict(changed_pixels=int((candidate!=initial).sum()),
                        added_pixels=int(((candidate>0)&(initial==0)).sum()),
                        removed_pixels=int(((candidate==0)&(initial>0)).sum()))
            if any(actual[k]!=report['results'][method][k] for k in actual):
                raise ValueError('Cached output disagrees with original report')
            cached[method]=candidate
            masks[method]=band&write
            outputs[method]=constrain(initial,candidate,masks[method])
        output_dir.mkdir(parents=True)
        save_mask(output_dir/'boundary_band.tif',band,initial_src.profile)
        result=dict(status='updates_complete',source_run=str(run_dir),radius_pixels=radius,
                    boundary_definition='Union instance label transitions, four-neighbor; Chebyshev dilation; computed before ROI clipping',
                    crop_edge_protection_pixels=margin,ground_truth_used_for_updates=False,
                    training_independence_confirmed=False,
                    scope='Fixed BsiNet IDs; cached accepted second-stage updates; developer ROI; no new SAM inference',
                    budget_warning='Same radius for both arms; boundary-intersected write budgets and retained changes need not match',
                    results={})
        for method,data in outputs.items():
            directory=output_dir/method
            directory.mkdir()
            save_labels(directory/'instance.tif',data,initial_src.profile)
            save_mask(directory/'allowed_write.tif',masks[method],initial_src.profile)
            result['results'][method]=dict(planned_write_pixels=report['write_pixels_per_method'],
                allowed_write_pixels=int(masks[method].sum()),
                unrestricted_changed_pixels=int((cached[method]!=initial).sum()),
                retained_changed_pixels=int((data!=initial).sum()))
        # Open GT only after both updates have been fixed and saved.
        if gt_path is not None:
            gt_path=Path(gt_path)
            if report.get('gt_sha256') and digest(gt_path)!=report['gt_sha256']:
                raise ValueError('GT differs from original evaluation')
            gt=stack.enter_context(rasterio.open(gt_path))
            aligned(baseline_src,gt,allow)
            if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
                raise ValueError('GT requires integer labels and NoData=None/0')
            target=gt.read(1,window=window)
            if np.any(target<0):
                raise ValueError('Negative GT labels')
            hashes[str(gt_path)]=digest(gt_path)
            rows=[]
            methods={'bsinet':baseline,'union':initial,
                     'conflict_unrestricted':cached['conflict'],'random_unrestricted':cached['random'],
                     'conflict_boundary':outputs['conflict'],'random_boundary':outputs['random']}
            for name,data in methods.items():
                counts=roi_counts(data,target,domain,params.get('boundary_tolerance',2))
                result.setdefault('evaluation',{})[name]=dict(scores(counts),counts=counts,
                    transitions_from_union=transitions(initial,data,target,domain))
                rows.append(dict(method=name,**scores(counts),**transitions(initial,data,target,domain)))
            with (output_dir/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if any(digest(path)!=expected for path,expected in hashes.items()):
            raise ValueError('Source changed during replay')
        result.update(status='complete',input_sha256=hashes,seconds=time.perf_counter()-start,
                      warning='Post-hoc development ablation, not independent test evidence. Cached output hashes are current snapshots.')
        (output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        lines=[f'Cached boundary replay: radius={radius}px; no SAM inference.']
        for method,entry in result['results'].items():
            lines.append(f"{method}: allowed={entry['allowed_write_pixels']}; changes={entry['unrestricted_changed_pixels']} -> {entry['retained_changed_pixels']}")
        for name,entry in result.get('evaluation',{}).items():
            change=entry['transitions_from_union']
            lines.append(f"{name}: IoU={entry['mask_iou']:.6f}, boundary_F1={entry['boundary_f1']:.6f}, instance_F1={entry['instance_f1']:.6f}, PQ={entry['pq']:.6f}, harmful_removals={change['removed_true_foreground']}, correct_removals={change['removed_false_foreground']}")
        text='\n'.join(lines)+'\n'
        (output_dir/'summary.txt').write_text(text,encoding='utf-8')
        print(text)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path)
    parser.add_argument('--radius',type=int,default=2)
    args=parser.parse_args()
    run(args.run_dir,args.output_dir,args.gt,args.radius)


if __name__=='__main__':
    main()
