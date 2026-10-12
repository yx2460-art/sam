"""Strict pixel/grid alignment and explicit RGB conversion for experiments."""
from pathlib import Path

import numpy as np
import rasterio


def read_labels(path):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one-channel instance IDs, not a color map: {path}")
        labels = src.read(1)
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError(f"Expected nonnegative integer instance IDs: {path}")
        # Existing BsiNet exports mark background ID 0 as NoData. Treat that
        # convention as background, not as ignored pixels, in this experiment.
        if src.nodata not in (None, 0) or (src.nodata is None and np.any(src.read_masks(1) == 0)):
            raise ValueError(f"NoData requires an explicit evaluation policy: {path}")
        return labels, src.profile.copy()


def check_alignment(first, second, first_path, second_path):
    if (first['height'], first['width']) != (second['height'], second['width']):
        raise ValueError(f"Shape mismatch: {first_path} / {second_path}; do not resize labels")
    if first.get('crs') is not None and second.get('crs') is not None:
        if first['crs'] != second['crs'] or not first['transform'].almost_equals(second['transform']):
            raise ValueError(f"Geospatial grid mismatch: {first_path} / {second_path}")


def read_rgb(path, bands, value_range):
    with rasterio.open(path) as src:
        if max(bands) > src.count or min(bands) < 1 or len(set(bands)) != 3:
            raise ValueError(f"Three distinct RGB band indices required: {path}")
        array = src.read(bands)
        if np.any(src.read_masks(bands) == 0):
            raise ValueError(f"Image contains NoData; select a fully valid tile: {path}")
        profile = src.profile.copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Image contains non-finite values: {path}")
    if value_range is None:
        if array.dtype != np.uint8:
            raise ValueError(f"Non-UInt8 image needs --rgb-range LOW HIGH: {path}")
        rgb = array
    else:
        low, high = value_range
        if high <= low:
            raise ValueError("--rgb-range HIGH must exceed LOW")
        rgb = np.clip((array.astype(np.float32) - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
    return np.moveaxis(rgb, 0, -1), profile


def save_labels(path, labels, profile):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(driver='GTiff', count=1, dtype='uint32', nodata=None, compress='deflate')
    for key in ('photometric', 'interleave'):
        profile.pop(key, None)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(labels.astype(np.uint32), 1)


def index_files(directory, suffix=''):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in {'.tif', '.tiff', '.png'}:
            continue
        if suffix and not path.stem.endswith(suffix):
            continue
        key = path.stem[:-len(suffix)] if suffix else path.stem
        if key in result:
            raise ValueError(f"Duplicate sample ID {key} in {directory}")
        result[key] = path
    if not result:
        raise ValueError(f"No matching rasters in {directory}")
    return result
