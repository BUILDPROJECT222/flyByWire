"""Extract a declared MaleCNS visual patch; retain only measured internal edges."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'malecns'
TYPES = ['L1','L2','L3','L4','L5','Mi1','Mi4','Mi9','Tm1','Tm2','Tm3','Tm9',
         'T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']


def main():
    annotations = feather.read_table(DATA / 'annotations.feather').to_pylist()
    by_id = {int(r['bodyId']): r for r in annotations}
    selected = {}
    for r in annotations:
        q, s = r['assignedOlHex1'], r['assignedOlHex2']
        if (r['type'] in TYPES and str(r['instance']).endswith('_R') and
                q is not None and s is not None and max(abs(q-19), abs(s-20), abs((q-19)-(s-20))) <= 4):
            selected[int(r['bodyId'])] = r
    # Assign photoreceptors to a selected measured L1/2/3 target, not arbitrary pixels.
    lamina = {bid for bid, r in selected.items() if r['type'] in ['L1','L2','L3']}
    receptors = {bid for bid, r in by_id.items() if r['type'] == 'R1-R6'}
    reader = pa.ipc.open_file(pa.memory_map(str(DATA / 'edges.feather')))
    retina = {}
    inferred = {}
    downstream = {}
    candidates = {bid for bid,r in by_id.items() if r['type'] in TYPES and bid not in selected and str(r['instance']).endswith('_R')}
    initial_ids = list(selected)
    for k in range(reader.num_record_batches):
        b = reader.get_batch(k)
        pre, post, w = [b.column(i).to_numpy() for i in range(3)]
        mask = np.isin(post, list(lamina))
        for a, z, weight in zip(pre[mask], post[mask], w[mask]):
            if int(a) in receptors and (int(a) not in retina or weight > retina[int(a)][1]):
                retina[int(a)] = (int(z), int(weight))
        mask = np.isin(pre, initial_ids)
        for a,z,weight in zip(pre[mask],post[mask],w[mask]):
            if int(z) in candidates and by_id[int(z)]['assignedOlHex1'] is None:
                item=downstream.setdefault(int(z),[0,int(a),0])
                item[0]+=int(weight)
                if weight>item[2]: item[1:]=[int(a),int(weight)]
    for typ in TYPES:
        ranked=sorted([bid for bid in downstream if by_id[bid]['type']==typ],key=lambda bid:downstream[bid][0],reverse=True)[:61]
        for bid in ranked:
            selected[bid]=by_id[bid]
            inferred[bid]=downstream[bid][1]
    for bid in retina:
        selected[bid] = by_id[bid]
    ids = sorted(selected)
    index = {bid:i for i,bid in enumerate(ids)}
    sources, targets, weights = [], [], []
    for k in range(reader.num_record_batches):
        b = reader.get_batch(k)
        pre, post, w = [b.column(i).to_numpy() for i in range(3)]
        mask = np.isin(pre, ids) & np.isin(post, ids)
        sources.extend(index[int(x)] for x in pre[mask])
        targets.extend(index[int(x)] for x in post[mask])
        weights.extend(w[mask].tolist())
    nt = {int(r['body']):r for r in feather.read_table(DATA / 'neurotransmitters.feather').to_pylist()}
    nodes = []
    for bid in ids:
        r = selected[bid]
        ref = selected[retina[bid][0]] if bid in retina else selected[inferred[bid]] if bid in inferred else r
        typ = r['type']
        chem = nt.get(bid,{}).get('consensus_nt') or nt.get(bid,{}).get('predicted_nt') or 'unknown'
        group = 'retina' if typ=='R1-R6' else 'lamina' if typ.startswith('L') else 'motion' if typ.startswith(('T4','T5')) else 'medulla'
        nodes.append(dict(id=str(bid), type=typ, group=group, transmitter=chem,
                          inhibitory=chem in ['gaba','glutamate','histamine'],
                          hex=[ref['assignedOlHex1'],ref['assignedOlHex2']],
                          soma=r.get('somaLocation'), hex_inferred=bid in retina or bid in inferred,
                          retina_target=str(retina[bid][0]) if bid in retina else None))
    np.savez_compressed(DATA / 'visual-circuit.npz', source=np.array(sources,np.int32),
                        target=np.array(targets,np.int32), weight=np.array(weights,np.float32))
    hashes = {}
    for name in ['annotations','neurotransmitters','edges']:
        with (DATA/(name+'.feather')).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream,'sha256').hexdigest()
    manifest = dict(dataset='MaleCNS v1.0',source='https://male-cns.janelia.org/download/',
        license='CC-BY-4.0',nodes=len(nodes),edges=len(weights),contacts=int(sum(weights)),
        selection='Right optic-lobe patch: axial radius 4 around published hex (19,20); 20 declared visual types; upstream R1–R6 into L1/L2/L3; at most 61 unhexed cells per declared type selected by total incoming contacts from the patch. Unhexed positions inferred from strongest patch partner.',
        exclusions='All cells and connections outside the patch are excluded. Boundary inputs are not reconstructed.',
        dynamics='Modeled signed leaky-rate network; neurotransmitter-to-sign mapping is an approximation, not receptor physiology.',
        input_mapping='Raycast luminance to inferred R1–R6 hex coordinates via strongest measured L1/2/3 partner. Not calibrated fly vision.',
        output_mapping='Trainable artificial two-action readout; not biological motor-neuron semantics.',
        sha256=hashes,types=dict(Counter(n['type'] for n in nodes)))
    (DATA/'visual-circuit.json').write_text(json.dumps(dict(manifest=manifest,nodes=nodes),indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
