"""Retain the complete non-null-superclass MaleCNS graph, matching Neural Canvas's policy."""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy.sparse import coo_matrix
from .core import DATA

def main():
    rows=feather.read_table(DATA/'annotations.feather').to_pylist()
    rows=sorted((r for r in rows if r['superclass'] is not None),key=lambda r:r['bodyId'])
    ids=np.array([r['bodyId'] for r in rows]);n=len(ids)
    reader=pa.ipc.open_file(pa.memory_map(str(DATA/'edges.feather')))
    aa=[];bb=[];cc=[]
    for k in range(reader.num_record_batches):
        batch=reader.get_batch(k);a,b,c=[batch.column(i).to_numpy() for i in range(3)]
        x=np.searchsorted(ids,a);y=np.searchsorted(ids,b)
        mask=(x<n)&(y<n)&(ids[np.minimum(x,n-1)]==a)&(ids[np.minimum(y,n-1)]==b)
        aa.append(x[mask].astype(np.int32));bb.append(y[mask].astype(np.int32));cc.append(c[mask].astype(np.uint32))
    a=np.concatenate(aa);b=np.concatenate(bb);c=np.concatenate(cc);del aa,bb,cc
    graph=coo_matrix((c,(a,b)),shape=(n,n)).tocsr()
    nt={int(r['body']):(r['consensus_nt'] or r['predicted_nt'] or 'unknown').lower() for r in feather.read_table(DATA/'neurotransmitters.feather',columns=['body','consensus_nt','predicted_nt']).to_pylist()}
    nodes=[];sign=[]
    classes=sorted(set(r['superclass'] for r in rows));coords=np.full((n,4),9999,np.float32)
    for i,r in enumerate(rows):
        chem=nt.get(int(ids[i]),'unknown');sgn=-1 if chem in ['gaba','glutamate','histamine'] else 1 if chem in ['acetylcholine','dopamine','octopamine','serotonin'] else 0
        sign.append(sgn)
        node=dict(id=str(r['bodyId']),type=r['type'] or 'untyped',group=r['superclass'],side=r['somaSide'],transmitter=chem,sign=sgn,soma=r['somaLocation'])
        nodes.append(node)
        if r['somaLocation'] is not None:coords[i,:3]=r['somaLocation']
        coords[i,3]=classes.index(r['superclass'])
    valid=np.array([r['somaLocation'] is not None for r in rows]);xyz=coords[valid,:3].copy();lo=xyz.min(axis=0);hi=xyz.max(axis=0)
    coords[valid,:3]=(xyz-(lo+hi)/2)/max(hi-lo)*2
    coords.tofile(DATA/'full-positions.bin')
    np.savez(DATA/'full-spiking.npz',offsets=graph.indptr.astype(np.uint32),targets=graph.indices.astype(np.uint32),counts=graph.data.astype(np.uint32),sign=np.array(sign,np.int32))
    manifest=dict(dataset='MaleCNS v1.0',nodes=n,edges=int(graph.nnz),contacts=int(c.sum(dtype=np.uint64)),positions=int(valid.sum()),classes=classes,selection='All annotation rows with non-null superclass; all measured internal edges retained, no strength threshold.',source='https://male-cns.janelia.org/download/',license='CC BY 4.0',coordinates='Published soma coordinates, uniformly centered and scaled for display; missing positions omitted only from rendering.',bounds=[lo.tolist(),hi.tolist()],unknown_signs=int(np.count_nonzero(np.array(sign)==0)),dynamics='Approximate LIF: dt 0.1 ms, delay 1.8 ms, refractory 2.2 ms, rest -52 mV, threshold -45 mV. All cells share parameters, including normally graded visual cells. Transmitter signs are approximate. No learning.',source_sha256=json.loads((DATA/'visual-circuit.json').read_text())['manifest']['sha256'])
    (DATA/'full-brain.json').write_text(json.dumps(dict(manifest=manifest,nodes=nodes),separators=(',',':')))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
