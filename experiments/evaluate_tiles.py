"""Score complete window runs on a fixed mask; partial smoke runs never receive scores."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from rasterio.windows import Window

from .metrics import instance_counts, label_boundary, pixel_counts, scores


def roi_counts(pred,gt,mask,tolerance):
    if not mask.any():
        raise ValueError('No scoring pixels')
    counts=pixel_counts(np.where(mask,pred,0),np.where(mask,gt,0))
    counts.update(instance_counts(np.where(mask,pred,0),np.where(mask,gt,0)))
    # Do not score artificial ROI or crop edges as parcel boundaries.
    interior=ndimage.binary_erosion(mask,iterations=tolerance+1,border_value=0)
    pb,gb=label_boundary(pred),label_boundary(gt)
    size=2*tolerance+1
    counts.update(boundary_pred=int((pb&interior).sum()),boundary_gt=int((gb&interior).sum()),
                  boundary_matched_pred=int((pb&interior&ndimage.maximum_filter(gb,size=size,mode='constant')).sum()),
                  boundary_matched_gt=int((gb&interior&ndimage.maximum_filter(pb,size=size,mode='constant')).sum()))
    return counts


def evaluate_run(directory,gt_path,mask_path,allow,tolerance):
    from .run_tiles import aligned
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':
        raise ValueError('Only complete runs can be evaluated')
    if tolerance<0:
        raise ValueError('Negative boundary tolerance')
    with ExitStack() as stack:
        image=stack.enter_context(rasterio.open(manifest['parameters']['image']))
        gt=stack.enter_context(rasterio.open(gt_path))
        roi=stack.enter_context(rasterio.open(mask_path))
        base=stack.enter_context(rasterio.open(manifest['parameters']['prediction']))
        for src in (gt,roi,base):
            aligned(image,src,allow)
        if not np.issubdtype(np.dtype(gt.dtypes[0]),np.integer) or gt.nodata not in (None,0):
            raise ValueError('GT must contain integer instance IDs with NoData=None/0')
        x0,y0,x1,y1=manifest['crop_xyxy']
        window=Window(x0,y0,x1-x0,y1-y0)
        raw_mask=roi.read(1,window=window)
        selected=(raw_mask>0)&np.isfinite(raw_mask)&(roi.read_masks(1,window=window)>0)
        # A different mask (or one outside the processed crop) invalidates comparability.
        from .run_tiles import digest
        if digest(mask_path)!=manifest['hashes']['eval_mask']:
            raise ValueError('Scoring mask differs from inference manifest')
        if digest(manifest['parameters']['prediction'])!=manifest['hashes']['prediction']:
            raise ValueError('BsiNet baseline file changed since inference')
        with rasterio.open(directory/'count.tif') as src:
            count=src.read(1)
        if np.any(selected & (count==0)):
            raise ValueError('Selected ROI has uncovered pixels; do not silently discard them')
        target=gt.read(1,window=window)
        baseline=base.read(1,window=window)
        if np.any(target<0):
            raise ValueError('Negative GT IDs')
        with rasterio.open(directory/'instance.tif') as src:
            if src.shape!=selected.shape or not src.transform.almost_equals(image.window_transform(window)):
                raise ValueError('Output crop grid mismatch')
            prediction=src.read(1)
        summary=dict(scoring_pixels=int(selected.sum()),boundary_tolerance_pixels=tolerance,
                     instance_scope='IDs clipped to fixed ROI, not whole-parcel test metrics',
                     training_independence_confirmed=False,gt_path=str(gt_path),mask_path=str(mask_path),
                     boundary_scope='ROI interior eroded by tolerance+1; no artificial mask boundary',
                     warning='Development diagnostics only: ROI-cut instances can affect F1/PQ. '
                             'Do not interpret as complete-parcel or independent test performance.')
        for key,data in [('baseline',baseline),('sam',prediction)]:
            counts=roi_counts(data,target,selected,tolerance)
            summary[key]={**scores(counts),'counts':counts}
        summary['delta_sam_minus_baseline']={key:summary['sam'][key]-value
                                             for key,value in summary['baseline'].items() if key!='counts'}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--eval-mask',type=Path,required=True)
    p.add_argument('--allow-crs-mismatch',action='store_true')
    p.add_argument('--boundary-tolerance',type=int,default=2)
    a=p.parse_args()
    evaluate_run(a.run_dir,a.gt,a.eval_mask,a.allow_crs_mismatch,a.boundary_tolerance)


if __name__=='__main__':
    main()
