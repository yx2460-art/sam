"""Build a static offline audit report with deterministically selected visual examples."""
import argparse
import html
import json
import struct
import zlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .diagnose_regions import transitions
from .evaluate_tiles import roi_counts
from .metrics import label_boundary
from .run_tiles import aligned, digest


def save_png(path, rgb):
    """Write an RGB PNG using only the standard library and numpy."""
    data=np.ascontiguousarray(rgb,dtype=np.uint8)
    height,width,channels=data.shape
    if channels!=3:
        raise ValueError('Expected RGB')
    def chunk(kind,payload):
        return struct.pack('!I',len(payload))+kind+payload+struct.pack('!I',zlib.crc32(kind+payload)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in data)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',width,height,8,2,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def select_cases(harmful, correct, domain, size, per_kind):
    if size<=0 or per_kind<=0:
        raise ValueError('Case size and count must be positive')
    candidates=[]
    for y in range(0,domain.shape[0],size):
        for x in range(0,domain.shape[1],size):
            patch=np.s_[y:y+size,x:x+size]
            if domain[patch].any():
                candidates.append(dict(x=x,y=y,width=min(size,domain.shape[1]-x),
                    height=min(size,domain.shape[0]-y),harmful=int(harmful[patch].sum()),
                    correct=int(correct[patch].sum())))
    selected=[]
    for kind in ('harmful','correct'):
        ranked=sorted((r for r in candidates if r[kind]>0),key=lambda r:(-r[kind],r['y'],r['x']))
        selected.extend(dict(r,selected_for=kind) for r in ranked[:per_kind])
    return selected


def overlay(rgb, labels, domain):
    image=rgb.copy()
    image[~domain]=(image[~domain].astype(np.float32)*0.2).astype(np.uint8)
    image[label_boundary(labels)&domain]=[255,255,0]
    return image


def build(boundary_dir,gt_path,output_dir,size=256,per_kind=2):
    boundary_dir,gt_path,output_dir=map(lambda p:Path(p).resolve(),(boundary_dir,gt_path,output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Use a new directory: {output_dir}')
    if size<=0 or per_kind<=0:
        raise ValueError('Case size and count must be positive')
    report_path=boundary_dir/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status')!='complete' or 'evaluation' not in report:
        raise ValueError('Expected completed, evaluated boundary replay')
    region_dir=Path(report['source_run'])
    original=json.loads((region_dir/'report.json').read_text(encoding='utf-8'))
    union_dir=Path(original['source_union'])
    manifest=json.loads((union_dir/'manifest.json').read_text(encoding='utf-8'))
    for parent in (boundary_dir,region_dir,union_dir,Path(manifest['source_run'])):
        if output_dir==parent.resolve() or parent.resolve() in output_dir.parents:
            raise ValueError('Output must be outside source runs')
    for path,expected in report['input_sha256'].items():
        if digest(path)!=expected:
            raise ValueError(f'Changed replay input: {path}')
    if digest(gt_path)!=original.get('gt_sha256'):
        raise ValueError('GT differs from region evaluation')
    params=manifest['parameters']
    paths={'bsinet':Path(params['prediction']),'union':union_dir/'instance.tif',
           'conflict_unrestricted':region_dir/'conflict'/'instance.tif',
           'random_unrestricted':region_dir/'random'/'instance.tif',
           'conflict_boundary':boundary_dir/'conflict'/'instance.tif',
           'random_boundary':boundary_dir/'random'/'instance.tif'}
    inputs=[report_path,region_dir/'report.json',union_dir/'manifest.json',gt_path,
            Path(params['image']),Path(params['eval_mask']),*paths.values()]
    hashes={str(p):digest(p) for p in inputs}
    allow=params.get('allow_crs_mismatch',False)
    x0,y0,x1,y1=manifest['crop_xyxy']
    window=Window(x0,y0,x1-x0,y1-y0)
    case_rows=[]
    with ExitStack() as stack:
        base=stack.enter_context(rasterio.open(paths['bsinet']))
        union_src=stack.enter_context(rasterio.open(paths['union']))
        gt=stack.enter_context(rasterio.open(gt_path))
        roi=stack.enter_context(rasterio.open(params['eval_mask']))
        image=stack.enter_context(rasterio.open(params['image']))
        aligned(base,gt,allow)
        aligned(base,roi,allow)
        if (image.shape!=base.shape or not image.transform.almost_equals(base.transform) or
                (not allow and image.crs!=base.crs)):
            raise ValueError('RGB source grid mismatch')
        if union_src.shape!=(y1-y0,x1-x0) or not union_src.transform.almost_equals(base.window_transform(window)):
            raise ValueError('Union crop grid mismatch')
        raw=roi.read(1,window=window)
        domain=(raw>0)&np.isfinite(raw)&(roi.read_masks(1,window=window)>0)
        target=gt.read(1,window=window)
        arrays={}
        for name,path in paths.items():
            src=stack.enter_context(rasterio.open(path))
            if name=='bsinet':
                data=src.read(1,window=window)
            else:
                aligned(union_src,src,allow)
                data=src.read(1)
            counts=roi_counts(data,target,domain,params.get('boundary_tolerance',2))
            entry=report['evaluation'][name]
            if counts!=entry['counts']:
                raise ValueError(f'{name}: cached raster no longer matches evaluated counts')
            arrays[name]=data
        initial=arrays['union']
        for method in ('conflict','random'):
            after=arrays[method+'_boundary']
            removed=(initial>0)&(after==0)&domain
            harmful=removed&(target>0)
            correct=removed&(target==0)
            for case in select_cases(harmful,correct,domain,size,per_kind):
                case_rows.append(dict(case,method=method))
        output_dir.mkdir(parents=True)
        panels=[]
        for index,case in enumerate(case_rows):
            x,y,w,h=(case[k] for k in ('x','y','width','height'))
            patch=np.s_[y:y+h,x:x+w]
            rgb=image.read(params.get('rgb_bands',[1,2,3]),window=Window(x0+x,y0+y,w,h))
            valid=np.all(image.read_masks(params.get('rgb_bands',[1,2,3]),window=Window(x0+x,y0+y,w,h))>0,axis=0)
            if params.get('rgb_range'):
                low,high=params['rgb_range']
                rgb=np.clip((rgb.astype(np.float32)-low)*255/(high-low),0,255)
            elif rgb.dtype!=np.uint8:
                raise ValueError('Non-uint8 RGB requires recorded rgb_range')
            rgb=np.moveaxis(rgb,0,-1).astype(np.uint8)
            rgb[~valid]=0
            local_domain=domain[patch]
            after=arrays[case['method']+'_boundary'][patch]
            before=initial[patch]
            truth=target[patch]
            changes=rgb.copy()
            changes[~local_domain]=(changes[~local_domain]*0.2).astype(np.uint8)
            removed=(before>0)&(after==0)&local_domain
            added=(before==0)&(after>0)&local_domain
            changes[removed&(truth>0)]=[255,0,0]
            changes[removed&(truth==0)]=[0,255,0]
            changes[added&(truth>0)]=[0,255,255]
            changes[added&(truth==0)]=[255,0,255]
            visuals=[('RGB',rgb),('GT',overlay(rgb,truth,local_domain)),
                     ('union',overlay(rgb,before,local_domain)),
                     ('unrestricted',overlay(rgb,arrays[case['method']+'_unrestricted'][patch],local_domain)),
                     ('boundary constrained',overlay(rgb,after,local_domain)),('changes from union',changes)]
            cards=[]
            for number,(title,picture) in enumerate(visuals):
                filename=f'case_{index:02d}_{number}.png'
                save_png(output_dir/filename,picture)
                cards.append(f'<figure><img src="{filename}" alt="{title}"><figcaption>{title}</figcaption></figure>')
            case.update(x_full=x0+x,y_full=y0+y,
                        transitions=transitions(before,after,truth,local_domain))
            panels.append(f"<h2>{case['method']} / top {case['selected_for']} example; x={x0+x}, y={y0+y}</h2>"+
                          '<div class="panels">'+''.join(cards)+'</div>'+f'<pre>{html.escape(json.dumps(case["transitions"]))}</pre>')
    columns=('mask_iou','boundary_f1','instance_f1','pq')
    table='<table><tr><th>method</th>'+''.join(f'<th>{key}</th>' for key in columns)+'</tr>'
    lines=['Development ROI report; no new inference; training independence unverified.']
    for name,entry in report['evaluation'].items():
        table+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{entry[k]:.6f}</td>' for k in columns)+'</tr>'
        lines.append(name+': '+', '.join(f'{k}={entry[k]:.6f}' for k in columns))
    table+='</table>'
    warning=('Post-hoc development examples selected using GT: fixed crop-grid tiles ranked by harmful/correct deletion counts, '
             'ties by y then x; up to the requested number per category and arm. These are extremes, not representative random samples. '
             'The same tile may appear in both categories. No examples when the category has no events. '
             'Yellow: instance boundaries; darkened: outside scoring ROI. Changes: red=harmful deletion, green=correct deletion, '
             'cyan=correct addition, magenta=false addition. GT is for analysis only; fixed IDs and ROI-clipped metrics remain limitations.')
    page='<!doctype html><meta charset="utf-8"><title>Region experiment audit</title><style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}.panels{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:256px;image-rendering:pixelated}pre{white-space:pre-wrap}</style>'
    page+='<h1>Region experiment audit</h1><p>'+html.escape(warning)+'</p>'+table+''.join(panels)
    if any(digest(path)!=value for path,value in hashes.items()):
        raise ValueError('Source changed during report generation')
    (output_dir/'index.html').write_text(page,encoding='utf-8')
    (output_dir/'cases.json').write_text(json.dumps(dict(selection='GT-based post-hoc extreme examples',
        size=size,per_kind=per_kind,cases=case_rows,input_sha256=hashes),indent=2),encoding='utf-8')
    text='\n'.join(lines)+f'\nVisual examples: {len(case_rows)}; open index.html in a browser.\n'
    (output_dir/'summary.txt').write_text(text,encoding='utf-8')
    print(text)
    return case_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--case-size',type=int,default=256)
    parser.add_argument('--per-kind',type=int,default=2)
    args=parser.parse_args()
    build(args.boundary_dir,args.gt,args.output_dir,args.case_size,args.per_kind)


if __name__=='__main__':
    main()
