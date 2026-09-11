"""Bounded parameter search with separated fit, selection, and untouched test phases."""
import json,time,shutil
from datetime import datetime,timezone
import numpy as np
from .core import DATA,ROOT
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .vision_experiment import stimulus
from .validate_full import sha

def main():
    out=ROOT/'experiments'/('visual-fit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True);(out/'source').mkdir()
    source_files=['lab/fit_visual_dynamics.py','lab/graded_engine.py','lab/shaders/full-graded.wgsl','lab/full_vision.py','lab/vision_experiment.py']
    for name in source_files:shutil.copy2(ROOT/name,out/'source'/name.replace('/','__'))
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes'];n=len(nodes)
    types=np.array([x['type'] for x in nodes])
    families={'L1':np.flatnonzero(types=='L1'),'L2':np.flatnonzero(types=='L2'),
              'T4':np.flatnonzero(np.char.startswith(types,'T4')),'T5':np.flatnonzero(np.char.startswith(types,'T5'))}
    outputs={'T4_T5':np.concatenate([families['T4'],families['T5']]),
             'descending':np.array([i for i,x in enumerate(nodes) if x['group']=='descending_neuron'])}
    encoder=RetinaEncoder();base_bias=np.full(n,.5,np.float32);base_bias[types=='R1-R6']=0
    with np.load(DATA/'full-graded.npz') as a:graph={k:a[k] for k in a.files}
    engine=GradedEngine(graph,base_bias)
    rng=np.random.default_rng(611);candidates=[{f:dict(tau_ms=50.,bias=.5) for f in families}]
    for _ in range(6):
        candidates.append({f:dict(tau_ms=float(np.exp(rng.uniform(np.log(10),np.log(100)))),
                                 bias=float(rng.uniform(0,.8))) for f in families})
    report=dict(scope='Exploratory eight-parameter full-graph visual fit, not a flight policy',
                train_seeds=list(range(500,506)),selection_seeds=list(range(600,604)),test_seeds=list(range(700,720)),
                candidate_seed=611,candidates=candidates,objective='T4/T5 selection accuracy; earliest candidate wins ties',
                parameter_bounds=dict(tau_ms=[10,100],bias=[0,.8]),candidate_results=[],test={},
                graph_sha256=sha(DATA/'full-graded.npz'),encoder_sha256=sha(DATA/'full-retina.json'),
                source_hashes={name:sha(ROOT/name) for name in source_files})
    def configure(parameters):
        bias=base_bias.copy();tau=np.full(n,50,np.float32)
        for family,indices in families.items():
            bias[indices]=parameters[family]['bias'];tau[indices]=parameters[family]['tau_ms']
        engine.configure(bias,tau)
        return engine.batch(np.zeros(n,np.float32),200)
    def episode(frames,baseline):
        engine.reset(baseline);states={s:[] for s in outputs}
        for image in frames:
            v=engine.batch(encoder.encode(image)/120,4)
            if not np.all(np.isfinite(v)):raise RuntimeError('Nonfinite voltage')
            for s,ids in outputs.items():states[s].append(v[ids])
        return {s:np.concatenate([part.mean(0) for part in np.array_split(np.array(values),3)]) for s,values in states.items()}
    def collect(seeds,baseline,condition='intact'):
        data={s:[] for s in outputs}
        for seed in seeds:
            for label in [-1,1]:
                frames=stimulus(seed,label)
                if condition=='shuffled':frames=frames[np.random.default_rng(seed+2000).permutation(12)]
                elif condition=='input_off':frames=np.zeros_like(frames)
                features=episode(frames,baseline)
                for s in data:data[s].append(features[s])
        return {s:np.array(v) for s,v in data.items()}
    def fit(X,y):
        mean=X.mean(0);scale=np.maximum(X.std(0),1e-4);z=(X-mean)/scale
        w=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y)
        return dict(mean=mean,scale=scale,weights=w)
    def scores(X,c):return ((X-c['mean'])/c['scale'])@c['weights']
    checkpoints=[]
    try:
        y=np.tile([-1,1],6);validation_y=np.tile([-1,1],4)
        for index,parameters in enumerate(candidates):
            baseline=configure(parameters)
            train=collect(report['train_seeds'],baseline)
            validation=collect(report['selection_seeds'],baseline)
            models={s:fit(train[s],y) for s in outputs};checkpoints.append(models)
            entry=dict(index=index,selection={})
            for s in outputs:
                value=scores(validation[s],models[s])
                entry['selection'][s]=dict(accuracy=float(np.mean(np.where(value>=0,1,-1)==validation_y)),scores=value.tolist())
                np.savez_compressed(out/f'candidate-{index}-{s}.npz',**models[s],train_features=train[s],
                                    selection_features=validation[s],indices=outputs[s])
            report['candidate_results'].append(entry)
            print('candidate',index,'selection:',{s:v['accuracy'] for s,v in entry['selection'].items()},flush=True)
        chosen=max(range(len(candidates)),key=lambda i:report['candidate_results'][i]['selection']['T4_T5']['accuracy'])
        report['selected']=chosen
        (out/'selection.json').write_text(json.dumps(report,indent=2))
        # No test data has been generated or evaluated before the selection checkpoint.
        truth=np.tile([-1,1],20)
        for index in sorted(set([0,chosen])):
            baseline=configure(candidates[index]);report['test'][str(index)]={}
            for condition in ['intact','shuffled','input_off']:
                if condition=='input_off':
                    one=episode(np.zeros((12,48,64),np.float32),baseline)
                    data={s:np.tile(v,(40,1)) for s,v in one.items()}
                else:data=collect(report['test_seeds'],baseline,condition)
                results={}
                for s in outputs:
                    value=scores(data[s],checkpoints[index][s]);correct=np.where(value>=0,1,-1)==truth
                    pairs=correct.reshape(20,2).mean(1)
                    draws=np.random.default_rng(911).integers(0,20,(10000,20))
                    results[s]=dict(accuracy=float(correct.mean()),scores=value.tolist(),
                        phase_bootstrap_95=np.quantile(pairs[draws].mean(1),[.025,.975]).tolist())
                report['test'][str(index)][condition]=results
                np.savez_compressed(out/f'test-{index}-{condition}.npz',**data,labels=truth)
                print('test',index,condition,{s:v['accuracy'] for s,v in results.items()},flush=True)
            # Paired flash against dark, all neurons retained for stage response inspection.
            traces=[]
            for light in [False,True]:
                engine.reset(baseline);trace=[]
                for t in range(60):
                    image=np.full((48,64),255 if light and 20<=t<40 else 0)
                    v=engine.batch(encoder.encode(image)/120,4)
                    trace.append([float(v[ids].mean()) for ids in list(families.values())+list(outputs.values())])
                traces.append(trace)
            np.savez_compressed(out/f'flash-{index}.npz',dark=traces[0],flash=traces[1],
                                stages=np.array(list(families)+list(outputs)))
        report['status']='completed; evaluate held-out results before any sensory qualification'
    except Exception as exc:
        report['error']=str(exc);raise
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)
    print('Selected',chosen,candidates[chosen])

if __name__=='__main__':main()
