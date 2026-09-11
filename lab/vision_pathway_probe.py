"""Exploratory background-drive sweep; no training or test-set selection."""
import json,time
from datetime import datetime,timezone
import numpy as np
from .core import DATA,ROOT
from .full_engine import FullEngine
from .full_vision import RetinaEncoder
from .validate_full import sha

def main():
    out=ROOT/'experiments'/('visual-pathway-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes']
    lamina=np.array([i for i,n in enumerate(nodes) if n['type'] in ['L1','L2','L3']])
    dn=np.array([i for i,n in enumerate(nodes) if n['group']=='descending_neuron'])
    encoder=RetinaEncoder();engine=FullEngine(np.load(DATA/'full-spiking.npz'))
    report=dict(scope='Exploratory pathway diagnosis, not biological parameter fitting',
        graph_sha256=sha(DATA/'full-spiking.npz'),encoder_sha256=sha(DATA/'full-retina.json'),
        source_sha256=sha(ROOT/'lab/vision_pathway_probe.py'),conditions=[])
    try:
        for background in [0,5,20,80]:
            for luminance in [0,255]:
                engine.reset();rates=encoder.encode(np.full((48,64),luminance));rates[lamina]=background
                snapshots=[]
                for _ in range(30):snapshots.append(engine.batch(rates,seed=87))
                counts=np.array(snapshots)
                name=f'background-{background}-light-{luminance}'
                ids=np.nonzero(counts)
                np.savez_compressed(out/(name+'.npz'),spikes=np.column_stack([*ids,counts[ids]]).astype(np.uint32))
                result=dict(background_hz=background,luminance=luminance,neural_ms=300,
                    retinal_spikes=int(counts[:,encoder.indices].sum()),
                    lamina_spikes=int(counts[:,lamina].sum()),
                    descending_spikes=int(counts[:,dn].sum()),all_spikes=int(counts.sum()))
                report['conditions'].append(result);print(result,flush=True)
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)

if __name__=='__main__':main()
