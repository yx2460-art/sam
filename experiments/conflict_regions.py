"""Ground-truth-free conflict selection and deterministic matched random controls."""
import numpy as np
from scipy import ndimage


def plan_regions(conflict, domain, core_size, buffer, min_pixels, max_regions):
    if core_size<=0 or buffer<0 or min_pixels<=0 or max_regions<=0:
        raise ValueError('Invalid region planning parameters')
    # Dilation merges nearby disagreements; ROI is never expanded for scoring.
    expanded=ndimage.binary_dilation(conflict,iterations=buffer) if buffer else conflict.copy()
    expanded &= domain
    cc,n=ndimage.label(expanded,structure=np.ones((3,3)))
    candidates=[]
    for component,box in enumerate(ndimage.find_objects(cc),start=1):
        if box is None:
            continue
        ys,xs=box
        for y in range(ys.start,ys.stop,core_size):
            for x in range(xs.start,xs.stop,core_size):
                h,w=min(core_size,ys.stop-y),min(core_size,xs.stop-x)
                mask=(cc[y:y+h,x:x+w]==component)
                amount=int((mask&conflict[y:y+h,x:x+w]).sum())
                if amount>=min_pixels:
                    candidates.append(dict(x=x,y=y,mask=mask,conflict_pixels=amount))
    candidates.sort(key=lambda r:(-r['conflict_pixels'],r['y'],r['x']))
    return candidates[:max_regions],len(candidates)


def random_controls(regions, domain, forbidden, foreground, seed, attempts=4000):
    rng=np.random.default_rng(seed)
    occupied=np.zeros(domain.shape,dtype=bool)
    result=[]
    for region in regions:
        mask=region['mask']
        h,w=mask.shape
        if h>domain.shape[0] or w>domain.shape[1]:
            raise ValueError('Random control does not fit domain')
        src_fg=int((foreground[region['y']:region['y']+h,region['x']:region['x']+w]&mask).sum())
        found=None
        for _ in range(attempts):
            y=int(rng.integers(0,domain.shape[0]-h+1))
            x=int(rng.integers(0,domain.shape[1]-w+1))
            sl=np.s_[y:y+h,x:x+w]
            if np.any(mask & (~domain[sl] | forbidden[sl] | occupied[sl])):
                continue
            fg=int((foreground[sl]&mask).sum())
            # Avoid spending the control budget on empty background. This uses
            # the BsiNet prediction, never GT. Match within 20% of mask area.
            if abs(fg-src_fg)>max(1,round(mask.sum()*0.2)):
                continue
            found=dict(x=x,y=y,mask=mask.copy(),matched_to=len(result),
                       bsinet_foreground_pixels=fg)
            occupied[sl] |= mask
            break
        if found is None:
            raise ValueError('Cannot place an equally sized random control outside conflict regions. '
                             'No inference performed. Inspect plan or explicitly reduce --max-regions; '
                             'do not treat unequal budgets as a matched comparison.')
        result.append(found)
    return result


def region_mask(regions,shape):
    out=np.zeros(shape,dtype=bool)
    for region in regions:
        y,x=region['y'],region['x']
        h,w=region['mask'].shape
        out[y:y+h,x:x+w] |= region['mask']
    return out
