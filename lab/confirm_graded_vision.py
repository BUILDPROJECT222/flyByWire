"""Frozen-model confirmation on fresh phases; no refitting or parameter search."""
import json,time
from datetime import datetime,timezone
import numpy as np
from .core import DATA,ROOT
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .vision_experiment import stimulus
from .validate_full import sha

def main():
    source=ROOT/'experiments/graded-vision-20260910T132410836064Z'
    out=ROOT/'experiments'/('graded-confirmation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes'];encoder=RetinaEncoder()
    bias=np.array([0 if n['type']=='R1-R6' else .5 for n in nodes],np.float32)
    with np.load(DATA/'full-graded.npz') as f:graph={k:f[k] for k in f.files}
    checkpoints={}
    for stage in ['retina','lamina','T4_T5','descending']:
        with np.load(source/(stage+'-readout.npz')) as f:checkpoints[stage]={k:f[k] for k in f.files}
    engine=GradedEngine(graph,bias)
    report=dict(source=str(source.relative_to(ROOT)),seeds=list(range(400,420)),trial_results=[],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [
            DATA/'full-graded.npz',DATA/'full-retina.json',ROOT/'lab/graded_engine.py',
            ROOT/'lab/shaders/full-graded.wgsl',ROOT/'lab/confirm_graded_vision.py']+
            [source/(s+'-readout.npz') for s in checkpoints]})
    try:
        baseline=engine.batch(np.zeros(len(nodes),np.float32),200)
        for seed in report['seeds']:
            for direction in [-1,1]:
                original=stimulus(seed,direction)
                for condition in ['intact','shuffled']:
                    frames=original if condition=='intact' else original[np.random.default_rng(seed+2000).permutation(12)]
                    engine.reset(baseline);values={s:[] for s in checkpoints}
                    for frame in frames:
                        v=engine.batch(encoder.encode(frame)/120,4)
                        for s,c in checkpoints.items():values[s].append(v[c['indices']])
                    predictions={};scores={}
                    for s,c in checkpoints.items():
                        feature=np.concatenate([x.mean(0) for x in np.array_split(np.array(values[s]),3)])
                        score=float(((feature-c['mean'])/c['scale'])@c['weights'])
                        predictions[s]=1 if score>=0 else -1;scores[s]=score
                    report['trial_results'].append(dict(seed=seed,label=direction,condition=condition,predictions=predictions,scores=scores))
            print('phase',seed,'complete',flush=True)
        rng=np.random.default_rng(90210);sample=rng.integers(0,20,size=(10000,20));summary={}
        for stage in checkpoints:
            paired={}
            for condition in ['intact','shuffled']:
                paired[condition]=np.array([np.mean([x['predictions'][stage]==x['label'] for x in report['trial_results']
                    if x['seed']==seed and x['condition']==condition]) for seed in report['seeds']])
            acc=paired['intact'];gap=acc-paired['shuffled']
            summary[stage]=dict(accuracy=float(acc.mean()),shuffle_accuracy=float(paired['shuffled'].mean()),
                phase_bootstrap_95=np.quantile(acc[sample].mean(1),[.025,.975]).tolist(),
                paired_shuffle_gap_95=np.quantile(gap[sample].mean(1),[.025,.975]).tolist())
        report['summary']=summary
        report['interpretation']='40 intact clips and 40 shuffled controls; 20 independent starting phases. Paired percentile bootstrap over phases, 10000 resamples. Fixed pilot models; no statistical motion-selectivity claim from eight pilot clips.'
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
