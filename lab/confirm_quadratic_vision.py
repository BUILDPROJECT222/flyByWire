"""Fresh confirmation of a frozen quadratic T4/T5 readout on the default full graph."""
import json,shutil
from datetime import datetime,timezone
import numpy as np
from .core import ROOT,DATA
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .vision_experiment import stimulus
from .validate_full import sha

def main():
    parent=ROOT/'experiments/visual-fit-20260910T133833056081Z'
    out=ROOT/'experiments'/('quadratic-confirmation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes']
    with np.load(parent/'quadratic-0-T4_T5.npz') as f:model={k:f[k] for k in f.files}
    with np.load(DATA/'full-graded.npz') as f:graph={k:f[k] for k in f.files}
    bias=np.array([0 if n['type']=='R1-R6' else .5 for n in nodes],np.float32)
    engine=GradedEngine(graph,bias);encoder=RetinaEncoder()
    report=dict(source=str(parent.relative_to(ROOT)),model='Default full graded network + frozen quadratic T4/T5 readout',
                seeds=list(range(1000,1020)),trials=[],conditions=['intact','shuffled','slow_coarse','fast_fine'],
                note='Fresh phases; slow_coarse uses frequency 2, phase increment .25, contrast 60; fast_fine uses frequency 4, increment .5, contrast 100. All 12 frames. No tuning.',
                hashes={str(p.relative_to(ROOT)):sha(p) for p in [parent/'quadratic-0-T4_T5.npz',
                DATA/'full-graded.npz',DATA/'full-retina.json',ROOT/'lab/graded_engine.py',
                ROOT/'lab/shaders/full-graded.wgsl',ROOT/'lab/confirm_quadratic_vision.py']})
    try:
        baseline=engine.batch(np.zeros(len(nodes),np.float32),200);features=[]
        for seed in report['seeds']:
            for label in [-1,1]:
                for condition in report['conditions']:
                    frames=stimulus(seed,label)
                    if condition=='shuffled':frames=frames[np.random.default_rng(seed+2000).permutation(12)]
                    elif condition in ['slow_coarse','fast_fine']:
                        frequency,speed,contrast=(2,.25,60) if condition=='slow_coarse' else (4,.5,100)
                        phase=np.random.default_rng(seed).uniform(0,2*np.pi)
                        x=np.linspace(0,1,64)[None,:];y=np.ones((48,1))
                        frames=np.array([127.5+contrast*np.sin(2*np.pi*x*frequency+phase-label*t*speed)*y for t in range(12)],np.float32)
                    engine.reset(baseline);values=[]
                    for frame in frames:
                        v=engine.batch(encoder.encode(frame)/120,4);values.append(v[model['indices']])
                    feature=np.concatenate([v.mean(0) for v in np.array_split(np.array(values),3)])
                    z=(feature-model['mean'])/model['scale']
                    score=float((1+z.astype(np.float64)@model['train'].astype(np.float64).T/len(z))**2@model['dual'])
                    features.append(feature)
                    report['trials'].append(dict(seed=seed,label=label,condition=condition,score=score,prediction=1 if score>=0 else -1))
            print('phase',seed,'complete',flush=True)
        report['summary']={}
        draws=np.random.default_rng(1901).integers(0,20,(10000,20))
        for condition in report['conditions']:
            paired=np.array([np.mean([t['prediction']==t['label'] for t in report['trials'] if t['seed']==s and t['condition']==condition]) for s in report['seeds']])
            report['summary'][condition]=dict(accuracy=float(paired.mean()),
                phase_bootstrap_95=np.quantile(paired[draws].mean(1),[.025,.975]).tolist())
        np.savez_compressed(out/'features.npz',features=np.array(features))
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2));print('Report:',out/'report.json',flush=True)
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
