"""Prepare complete annotated MaleCNS graph for backend comparisons, not a validated model."""
from pathlib import Path
import json,time
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy.sparse import coo_matrix,save_npz
root=Path(__file__).resolve().parents[1]; data=root/'data/malecns'
start=time.perf_counter()
annotations=feather.read_table(data/'annotations.feather',columns=['bodyId','status'])
mask=np.array(annotations['status'].to_pylist())=='Traced'
ids=np.sort(np.unique(annotations['bodyId'].to_numpy()[mask]))
reader=pa.ipc.open_file(pa.memory_map(str(data/'edges.feather')))
a=[];b=[];v=[]
for k in range(reader.num_record_batches):
    batch=reader.get_batch(k)
    pre,post,w=[batch.column(i).to_numpy() for i in range(3)]
    p=np.searchsorted(ids,pre);q=np.searchsorted(ids,post)
    mask=(p<len(ids))&(q<len(ids))
    mask &= (ids[np.minimum(p,len(ids)-1)]==pre)&(ids[np.minimum(q,len(ids)-1)]==post)
    a.append(p[mask].astype(np.int32));b.append(q[mask].astype(np.int32));v.append(w[mask].astype(np.float32))
    if k%500==0: print(f'Batch {k}/{reader.num_record_batches}',flush=True)
source=np.concatenate(a);target=np.concatenate(b);counts=np.concatenate(v)
del a,b,v
nt=feather.read_table(data/'neurotransmitters.feather',columns=['body','consensus_nt','predicted_nt']).to_pylist()
byid={int(r['body']):(r['consensus_nt'] or r['predicted_nt'] or '').lower() for r in nt};del nt
sign=np.array([-1 if byid.get(int(i)) in ['gaba','histamine','glutamate'] else 1 for i in ids],np.float32)
norm=np.bincount(target,weights=counts,minlength=len(ids))
weights=counts*sign[source]/np.maximum(1,norm[target])
matrix=coo_matrix((weights.astype(np.float32),(target,source)),shape=(len(ids),len(ids))).tocsr()
save_npz(data/'benchmark-full-csr.npz',matrix,compressed=False)
meta=dict(nodes=len(ids),edges=len(counts),csr_nonzero=int(matrix.nnz),contacts=int(counts.sum(dtype=np.float64)),preparation_seconds=time.perf_counter()-start,
          scope='All status=Traced neurons; glia/fragments/non-Traced records excluded; input-normalized approximate transmitter signs; synthetic drive for a backend benchmark',
          graph_bytes=matrix.indptr.nbytes+matrix.indices.nbytes+matrix.data.nbytes)
(data/'benchmark-full-csr.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
