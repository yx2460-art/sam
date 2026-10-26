"""Fixed-origin raster windows and stable-source-ID fusion for scale ablations."""
import numpy as np
from rasterio.windows import Window


def windows(width, height, size, stride, bounds):
    if size <= 0 or not 0 < stride <= size:
        raise ValueError('Require 0 < stride <= tile size')
    x0, y0, x1, y1 = bounds
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            w, h = min(size, width-x), min(size, height-y)
            if x < x1 and x+w > x0 and y < y1 and y+h > y0:
                yield Window(x, y, w, h)


def mask_bounds(src, context):
    left, top, right, bottom = src.width, src.height, 0, 0
    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        selected = (data > 0) & np.isfinite(data) & (src.read_masks(1, window=window) > 0)
        ys, xs = np.where(selected)
        if xs.size:
            left = min(left, int(window.col_off)+int(xs.min()))
            right = max(right, int(window.col_off)+int(xs.max())+1)
            top = min(top, int(window.row_off)+int(ys.min()))
            bottom = max(bottom, int(window.row_off)+int(ys.max())+1)
    if right <= left:
        raise ValueError('Evaluation mask is empty')
    return (max(0, left-context), max(0, top-context),
            min(src.width, right+context), min(src.height, bottom+context))


def source_ids(local, original):
    """Only the fixed shrink-only protocol is supported; never infer global ID by proximity."""
    out = np.zeros(local.shape, dtype=np.uint32)
    for label in np.unique(local[local > 0]):
        region = local == label
        ids = np.unique(original[region])
        if len(ids) != 1 or ids[0] == 0:
            raise ValueError('Local candidate must belong to exactly one nonzero source ID')
        if int(ids[0]) > np.iinfo(np.uint32).max:
            raise ValueError('Source ID exceeds UInt32')
        out[region] = ids[0]
    return out


def center_weights(height, width):
    y, x = np.indices((height, width))
    return np.minimum(np.minimum(x+1, width-x), np.minimum(y+1, height-y)).astype(np.float32)


def fuse_result(original, votes, count, center, method):
    if method == 'center':
        foreground = center
    elif method == 'union':
        foreground = votes > 0
    elif method == 'intersection':
        foreground = (count > 0) & (votes == count)
    elif method == 'majority':
        foreground = votes.astype(np.uint64)*2 > count  # ties are background
    else:
        raise ValueError(method)
    return np.where(foreground & (count > 0), original, 0).astype(np.uint32)
