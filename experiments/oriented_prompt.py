"""Adapt a human rotated rectangle to SAM's axis-aligned box interface.

Rotate the complete context, never clip the predicted mask to the prompt box.
All exported masks remain on the original raster grid.
"""
import numpy as np
from scipy import ndimage


def geometry(points,shape):
    p=np.asarray(points)
    if p.shape!=(3,2) or not np.issubdtype(p.dtype,np.number) or not np.isfinite(p).all():
        raise ValueError('Oriented box requires three finite pixel points')
    p=p.astype(float);edge=p[1]-p[0];length=float(np.linalg.norm(edge))
    if length<2: raise ValueError('Oriented box long edge must be at least two pixels')
    u=edge/length;v=np.array([-u[1],u[0]]);depth=float((p[2]-p[0])@v)
    if abs(depth)<2: raise ValueError('Oriented box width must be at least two pixels')
    corners=np.array([p[0],p[1],p[1]+depth*v,p[0]+depth*v])
    h,w=shape
    if np.any(corners<0) or np.any(corners[:,0]>w-1) or np.any(corners[:,1]>h-1):
        raise ValueError('Oriented box corners outside image context; redraw inside image')
    rotation=np.array([u,v])
    bounds=corners@rotation.T
    return dict(corners=corners,rotation=rotation,lower=bounds.min(axis=0),upper=bounds.max(axis=0))


def validate_prompts(box,shape,points,labels):
    g=geometry(box,shape)
    positive=np.asarray(points,dtype=float)[np.asarray(labels)==1]@g['rotation'].T
    if np.any(positive<g['lower']-1e-6) or np.any(positive>g['upper']+1e-6):
        raise ValueError('All positive points must lie inside the oriented target box')
    return g


class OrientedPredictor:
    """Resample RGB once, transform prompts, and restore SAM logits to original pixels."""
    def __init__(self,predictor,box):
        self.predictor=predictor;self.box=box;self.model=predictor.model

    def set_image(self,rgb):
        self.shape=rgb.shape[:2];g=geometry(self.box,self.shape);self.g=g
        h,w=self.shape
        frame=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])@g['rotation'].T
        self.offset=frame.min(axis=0)
        size=np.ceil(frame.max(axis=0)-self.offset).astype(int)+1
        if size.max()>8192 or np.prod(size)>32_000_000:
            raise ValueError('Rotated context too large; use a smaller fixed region')
        self.rotated_shape=(int(size[1]),int(size[0]))
        inverse=g['rotation'].T
        # scipy works in row/column order, while prompt coordinates use x/y.
        matrix=inverse[::-1,::-1];offset=(inverse@self.offset)[::-1]
        rotated=np.stack([ndimage.affine_transform(rgb[:,:,band],matrix,offset,
            output_shape=self.rotated_shape,order=1,mode='constant',cval=0,prefilter=False)
            for band in range(3)],axis=-1)
        self.predictor.set_image(rotated)
        self.prompt_box=np.r_[g['lower']-self.offset,g['upper']-self.offset].astype(np.float32)

    def predict(self,point_coords,point_labels,box=None,**kwargs):
        coords=np.asarray(point_coords)@self.g['rotation'].T-self.offset
        masks,quality,_=self.predictor.predict(point_coords=coords.astype(np.float32),
            point_labels=point_labels,box=self.prompt_box,**kwargs)
        restored=np.stack([ndimage.affine_transform(np.asarray(raw,dtype=float),
            self.g['rotation'][::-1,::-1],(-self.offset)[::-1],output_shape=self.shape,
            order=1,mode='constant',cval=-1e6,prefilter=False) for raw in masks])
        return restored,quality,None

    def audit(self):
        return dict(mode='rotated-image-box',input_points=self.box,
            original_corners=self.g['corners'].tolist(),rotation_xy=self.g['rotation'].tolist(),
            rotated_origin_xy=self.offset.tolist(),sam_box_xyxy=self.prompt_box.tolist(),
            rotated_shape=list(self.rotated_shape),mask_clipped_to_box=False,
            resampling='RGB bilinear; full-resolution logits bilinear back to original grid')
