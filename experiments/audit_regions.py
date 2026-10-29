"""Audit declared spatial split masks; never infer checkpoint training provenance."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from .run_tiles import aligned, digest


def run(config_path,output_dir):
    config_path,output_dir=Path(config_path).resolve(),Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Use a new directory: {output_dir}')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    radius=config.get('buffer_pixels',256)
    if type(radius) is not int or radius<0:
        raise ValueError('buffer_pixels must be a nonnegative integer')
    required=['reference','checkpoint','training_provenance','train_mask','validation_mask','test_mask']
    missing=[key for key in required if not config.get(key)]
    if config.get('all_training_footprints_declared') is not True:
        missing.append('all_training_footprints_declared=true (user attestation)')
    result=dict(status='pending_inputs',training_independence_confirmed=False,
        missing=missing,buffer_pixels=radius,
        warning='Spatial checks cannot recover actual training sources or certify independent data. Provenance is a user attestation; spatial masks alone are insufficient.',
        config_sha256=digest(config_path))
    if not missing:
        def resolve(value):
            path=Path(value)
            return path if path.is_absolute() else config_path.parent/path
        names=('train','validation','test')
        paths={name:resolve(config[name+'_mask']) for name in names}
        paths.update(reference=resolve(config['reference']),checkpoint=resolve(config['checkpoint']))
        hashes={str(path.resolve()):digest(path) for path in paths.values()}
        areas=dict.fromkeys(names,0)
        pairs=(('train','validation'),('train','test'),('validation','test'))
        overlap={a+'_'+b:0 for a,b in pairs}
        buffered={a+'_'+b:0 for a,b in pairs}
        with ExitStack() as stack:
            reference=stack.enter_context(rasterio.open(paths['reference']))
            sources={name:stack.enter_context(rasterio.open(paths[name])) for name in names}
            for src in sources.values():
                aligned(reference,src,False)
            def binary(src,window):
                raw=src.read(1,window=window,boundless=True,fill_value=0)
                if np.any(~np.isfinite(raw)) or np.any((raw!=0)&(raw!=1)):
                    raise ValueError('Split masks must be binary 0/1, not instance IDs')
                if np.any((raw>0)&(src.read_masks(1,window=window,boundless=True)==0)):
                    raise ValueError('Positive mask pixel marked invalid')
                return raw>0
            # Fixed blocks avoid holding all full-image masks in memory.
            for y in range(0,reference.height,512):
                for x in range(0,reference.width,512):
                    w,h=min(512,reference.width-x),min(512,reference.height-y)
                    window=Window(x,y,w,h)
                    masks={name:binary(src,window) for name,src in sources.items()}
                    for name in names:
                        areas[name]+=int(masks[name].sum())
                    near={}
                    halo=Window(x-radius,y-radius,w+2*radius,h+2*radius)
                    for name in ('train','validation'):
                        expanded=binary(sources[name],halo)
                        expanded=ndimage.maximum_filter(expanded,size=2*radius+1,mode='constant')
                        near[name]=expanded[radius:radius+h,radius:radius+w]
                    for a,b in pairs:
                        key=a+'_'+b
                        overlap[key]+=int((masks[a]&masks[b]).sum())
                        buffered[key]+=int((near[a]&masks[b]).sum())
        if any(digest(path)!=value for path,value in hashes.items()):
            raise ValueError('Source changed during split audit')
        passed=all(areas.values()) and not any(buffered.values())
        result.update(status='spatial_checks_passed' if passed else 'spatial_checks_failed',
            areas_pixels=areas,overlap_pixels=overlap,buffer_violations_pixels=buffered,
            distance='Chebyshev pixel distance <= buffer is a violation; exact CRS/grid required',
            training_provenance=config['training_provenance'],
            provenance_scope='user supplied, not independently verified',input_sha256=hashes)
    if digest(config_path)!=result['config_sha256']:
        raise ValueError('Configuration changed during audit')
    output_dir.mkdir(parents=True)
    (output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=json.dumps(result,indent=2)+'\n'
    (output_dir/'summary.txt').write_text(text,encoding='utf-8')
    print(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.config,args.output_dir)


if __name__=='__main__':
    main()
