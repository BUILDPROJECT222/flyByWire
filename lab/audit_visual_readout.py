"""Exploratory fixed quadratic readout: test whether linear decoding is the bottleneck."""
import json
import numpy as np
from .core import ROOT

def main():
    folder=ROOT/'experiments/visual-fit-20260910T133833056081Z'
    experiment=json.loads((folder/'report.json').read_text())
    y=np.tile([-1.,1.],len(experiment['train_seeds']));truth=np.tile([-1,1],len(experiment['test_seeds']))
    result=dict(scope='Post-hoc readout-capacity diagnostic; fresh confirmation required',
                source=str(folder.relative_to(ROOT)),kernel='(1 + standardized dot product / feature count)^2',
                regularization=.001,results={})
    for index in [0,experiment['selected']]:
        result['results'][str(index)]={}
        for stage in ['T4_T5','descending']:
            with np.load(folder/f'candidate-{index}-{stage}.npz') as checkpoint:
                x=(checkpoint['train_features']-checkpoint['mean'])/checkpoint['scale']
                mean=checkpoint['mean'];scale=checkpoint['scale']
            kernel=lambda a,b:(1+a.astype(np.float64)@b.astype(np.float64).T/x.shape[1])**2
            dual=np.linalg.solve(kernel(x,x)+np.eye(len(x))*.001,y)
            conditions={}
            for condition in ['intact','shuffled','input_off']:
                with np.load(folder/f'test-{index}-{condition}.npz') as f:test=(f[stage]-mean)/scale
                scores=kernel(test,x)@dual
                conditions[condition]=dict(accuracy=float(np.mean(np.where(scores>=0,1,-1)==truth)),scores=scores.tolist())
            result['results'][str(index)][stage]=conditions
            np.savez_compressed(folder/f'quadratic-{index}-{stage}.npz',train=x,mean=mean,scale=scale,
                                dual=dual,indices=np.load(folder/f'candidate-{index}-{stage}.npz')['indices'])
    (folder/'quadratic-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({i:{s:{c:v['accuracy'] for c,v in r.items()} for s,r in d.items()} for i,d in result['results'].items()},indent=2))

if __name__=='__main__':main()
