"""Engineered luminance-to-retina interface; not calibrated fly optics."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import DATA

VERSION = 'full-retina-luminance-v1'


class RetinaEncoder:
    def __init__(self, metadata=None):
        self.meta = metadata or json.loads((DATA/'full-retina.json').read_text())
        self.n = self.meta['nodes']
        self.indices = np.array([r['index'] for r in self.meta['receptors']], int)
        self.uv = np.array([r['uv'] for r in self.meta['receptors']], np.float32)

    def encode(self, image):
        """Accept already-upright RGB or grayscale; fixed 0..255 scale, no per-frame normalization."""
        a = np.asarray(image)
        if a.ndim == 3 and a.shape[2] == 3:
            a = a.astype(np.float32) @ np.array([.2126,.7152,.0722], np.float32)
        if a.ndim != 2 or min(a.shape) < 2 or not np.all(np.isfinite(a)):
            raise ValueError('Expected finite grayscale or RGB image at least 2x2')
        if np.any(a < 0) or np.any(a > 255):
            raise ValueError('Pixels must be in 0..255')
        y = self.uv[:,1]*(a.shape[0]-1)
        x = self.uv[:,0]*(a.shape[1]-1)
        x0=x.astype(int);y0=y.astype(int)
        x1=np.minimum(x0+1,a.shape[1]-1);y1=np.minimum(y0+1,a.shape[0]-1)
        dx=x-x0;dy=y-y0
        luminance=(a[y0,x0]*(1-dx)*(1-dy)+a[y0,x1]*dx*(1-dy)+
                   a[y1,x0]*(1-dx)*dy+a[y1,x1]*dx*dy)/255
        rates=np.zeros(self.n,np.float32)
        rates[self.indices]=luminance*120
        return rates


def prepare():
    import pyarrow.feather as feather
    rows=feather.read_table(DATA/'annotations.feather').to_pylist()
    by_id={str(r['bodyId']):r for r in rows}
    full=json.loads((DATA/'full-brain.json').read_text())
    nodes=full['nodes']
    with np.load(DATA/'full-spiking.npz') as archive:
        graph={key:archive[key] for key in ['offsets','targets','counts']}
    receptors=[];missing=[]
    for i,n in enumerate(nodes):
        if n['type']!='R1-R6':continue
        row=by_id[n['id']];side=str(row['instance'])[-1]
        candidates=[]
        for edge in range(graph['offsets'][i],graph['offsets'][i+1]):
            target=int(graph['targets'][edge]);r=by_id[nodes[target]['id']]
            if (r['type'] in ['L1','L2','L3'] and str(r['instance']).endswith('_'+side)
                and r['assignedOlHex1'] is not None and r['assignedOlHex2'] is not None):
                candidates.append((int(graph['counts'][edge]),-target,target,r))
        if not candidates:missing.append(n['id']);continue
        weight,_,target,r=max(candidates,key=lambda c:c[:2])
        q,s=r['assignedOlHex1'],r['assignedOlHex2']
        receptors.append(dict(index=i,id=n['id'],side=side,target=nodes[target]['id'],
            contacts=weight,hex=[q,s],xy=[q-s/2,s*np.sqrt(3)/2]))
    for side in ['L','R']:
        group=[r for r in receptors if r['side']==side]
        xy=np.array([r['xy'] for r in group])
        lo=xy.min(0);hi=xy.max(0)
        for r in group:
            uv=(np.array(r['xy'])-lo)/(hi-lo)
            if side=='L':uv[0]=1-uv[0]
            r['uv']=uv.tolist()
    meta=dict(version=VERSION,nodes=len(nodes),receptors=receptors,unmapped_ids=missing,
        mapping='Strongest same-side measured L1/L2/L3 partner with published hex; each eye fills the same image; left mirrored. Axial hex projected to 2D then independently normalized per eye. Engineered orientation and field of view, not calibrated optics.',
        input='Already clockwise-rotated upright 0..255 RGB/grayscale; fixed luminance to 0..120 Hz. No optical flow or task label input; unmapped receptors receive zero external drive.',
        source_sha256=full['manifest']['source_sha256'])
    (DATA/'full-retina.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(mapped=len(receptors),unmapped=len(missing),
                         sides={s:sum(r['side']==s for r in receptors) for s in ['L','R']})))


if __name__=='__main__':prepare()
