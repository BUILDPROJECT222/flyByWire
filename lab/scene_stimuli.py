"""Labeled artificial camera stimuli, independent of the frozen model."""
import numpy as np
from PIL import Image,ImageFilter

def texture(seed):
    rng=np.random.default_rng(seed)
    raw=rng.integers(0,256,(64,160),dtype=np.uint8)
    return np.asarray(Image.fromarray(raw).filter(ImageFilter.GaussianBlur(1.1)),np.float32)

def clip(seed,kind,direction=1,frames=12):
    rng=np.random.default_rng(seed);x=np.arange(64)[None,:];y=np.arange(48)[:,None]
    if kind in ['texture','stationary']:
        canvas=texture(seed);speed=0 if kind=='stationary' else direction*int(rng.integers(1,3))
        # Rightward image motion samples progressively further left in the canvas.
        return np.array([canvas[8:56,64-speed*t:128-speed*t] for t in range(frames)])
    if kind in ['on_edge','off_edge']:
        speed=float(rng.uniform(1.5,2.5));start=16 if direction>0 else 48
        result=[]
        for t in range(frames):
            boundary=start+direction*speed*t
            bright=(x<boundary) if kind=='on_edge' else (x>=boundary)
            result.append(np.broadcast_to(30+190*bright,(48,64)).astype(np.float32))
        return np.array(result)
    if kind=='looming':
        cx=float(rng.uniform(24,40));cy=float(rng.uniform(18,30))
        return np.array([np.where((x-cx)**2+(y-cy)**2<(2+t*1.1)**2,30,220) for t in range(frames)],np.float32)
    raise ValueError(kind)

def image_motion(frames):
    """Independent small horizontal-translation matcher; returns -1, 0, +1."""
    a=np.asarray(frames,np.float32)
    if a.ndim==4:a=a.mean(3)
    shifts=np.arange(-3,4);errors=[]
    for dx in shifts:
        errors.append(np.mean((a[1:,:,3:-3]-a[:-1,:,3-dx:a.shape[2]-3-dx])**2))
    chosen=int(shifts[np.argmin(errors)])
    return int(np.sign(chosen))
