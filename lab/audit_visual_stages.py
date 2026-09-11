"""Localize information loss using existing full-network count recordings."""
import json
from pathlib import Path
import numpy as np
from .core import DATA,ROOT

def main():
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes']
    groups={
        'retina':[i for i,n in enumerate(nodes) if n['type']=='R1-R6'],
        'lamina':[i for i,n in enumerate(nodes) if n['type'] in ['L1','L2','L3']],
        'T4_T5':[i for i,n in enumerate(nodes) if n['type'].startswith(('T4','T5'))],
        'visual_projection':[i for i,n in enumerate(nodes) if n['group']=='visual_projection'],
        'descending':[i for i,n in enumerate(nodes) if n['group']=='descending_neuron']}
    report={}
    for run in ['full-vision-20260910T131456086304Z','full-vision-20260910T131701141382Z']:
        folder=ROOT/'experiments'/run;meta=json.loads((folder/'report.json').read_text())
        data={}
        for stage,indices in groups.items():
            lookup=np.full(len(nodes),-1);lookup[indices]=np.arange(len(indices))
            def features(name):
                with np.load(folder/(name+'.npz')) as a:rows=a['spikes']
                local=lookup[rows[:,1]];mask=local>=0
                counts=np.zeros((3,len(indices)),np.float32)
                np.add.at(counts,(rows[mask,0]//8,local[mask]),rows[mask,2])
                return counts.flatten()/4  # mean of four 20 ms frame counts, matching original readout
            X=np.array([features(f'train-{s}-{d}') for s in meta['train_seeds'] for d in [-1,1]])
            y=np.tile([-1,1],len(meta['train_seeds']))
            mean=X.mean(0);scale=X.std(0);scale[scale<1]=1;z=(X-mean)/scale
            w=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y)
            results={}
            for condition,suffix in [('intact',''),('shuffled','-shuffled'),('disconnected','-disconnected')]:
                test=np.array([features(f'test-{s}-{d}'+suffix) for s in meta['test_seeds'] for d in [-1,1]])
                truth=np.tile([-1,1],len(meta['test_seeds']))
                scores=((test-mean)/scale)@w
                results[condition]=dict(accuracy=float(np.mean(np.where(scores>=0,1,-1)==truth)),
                    total_spikes=int(test.sum()*4),varying_features=int(np.count_nonzero(test.std(0)>0)))
            data[stage]=dict(neurons=len(indices),results=results)
        report[run]=data
    path=ROOT/'benchmarks/visual-stage-audit.json';path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
