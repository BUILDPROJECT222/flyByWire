"""Full-graph graded visual prototype; separate from the spiking UI."""
import json,time
from datetime import datetime,timezone
import numpy as np
from .core import ROOT,DATA
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .vision_experiment import stimulus
from .validate_full import sha

def main():
    out=ROOT/'experiments'/('graded-vision-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes']
    n=len(nodes);encoder=RetinaEncoder()
    stages={name:np.array([i for i,x in enumerate(nodes) if test(x)]) for name,test in {
        'retina':lambda x:x['type']=='R1-R6',
        'lamina':lambda x:x['type'] in ['L1','L2','L3'],
        'T4_T5':lambda x:x['type'].startswith(('T4','T5')),
        'descending':lambda x:x['group']=='descending_neuron'}.items()}
    bias=np.full(n,.5,np.float32);bias[stages['retina']]=0
    with np.load(DATA/'full-graded.npz') as a:graph={k:a[k] for k in a.files}
    engine=GradedEngine(graph,bias)
    report=dict(scope='Untrained dimensionless graded full-graph prototype; not a FlyVis reproduction or spike model.',
        nodes=n,edges=len(graph['weights']),dt_ms=5,tau_ms=50,gain=.8,
        bias='0 for R1-R6; 0.5 otherwise',weight_rule='Signed contacts divided by total measured incoming contacts, including unknown signs in denominator',
        train_seeds=list(range(200,208)),test_seeds=list(range(300,304)),trials=[],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'full-graded.npz',DATA/'full-retina.json',
            ROOT/'lab/graded_engine.py',ROOT/'lab/shaders/full-graded.wgsl',ROOT/'lab/graded_vision_experiment.py']})
    try:
        zero=np.zeros(n,np.float32);baseline=engine.batch(zero,200)
        def run(frames,name):
            engine.reset(baseline);values=[];times=[]
            for frame in frames:
                drive=encoder.encode(frame)/120
                t=time.perf_counter();v=engine.batch(drive,4);times.append((time.perf_counter()-t)*1000)
                if not np.all(np.isfinite(v)):raise RuntimeError('Nonfinite state')
                values.append(v)
            values=np.array(values)
            np.savez_compressed(out/(name+'.npz'),voltage=values,input_frames=frames,wall_ms=times)
            report['trials'].append(dict(name=name,neural_ms=len(frames)*20,
                p95_frame_ms=float(np.percentile(times,95)),max_abs_voltage=float(np.abs(values).max())))
            return {s:np.concatenate([part.mean(0) for part in np.array_split(values[:,ids],3)]) for s,ids in stages.items()}
        # A paired dark/bright/recovery protocol also tests propagation independently of classification.
        dark=np.zeros((60,48,64),np.float32)
        flash=dark.copy();flash[20:40]=255
        run(dark,'dark');run(flash,'flash')
        a=np.load(out/'flash.npz')['voltage'];b=np.load(out/'dark.npz')['voltage']
        report['flash_response']={s:dict(mean_abs_change=float(np.abs(a[:,ids]-b[:,ids]).mean()),
            peak_abs_change=float(np.abs(a[:,ids]-b[:,ids]).max()),
            light_mean_change=float((a[20:40,ids]-b[20:40,ids]).mean())) for s,ids in stages.items()}
        records={name:[] for name in ['train','test','shuffled','input_off']}
        for split,seeds in [('train',report['train_seeds']),('test',report['test_seeds'])]:
            for seed in seeds:
                for direction in [-1,1]:
                    frames=stimulus(seed,direction);name=f'{split}-{seed}-{direction}'
                    records[split].append(run(frames,name))
                    if split=='test':
                        order=np.random.default_rng(seed+2000).permutation(len(frames))
                        records['shuffled'].append(run(frames[order],name+'-shuffled'))
                        records['input_off'].append(run(np.zeros_like(frames),name+'-input-off'))
                print(split,seed,'complete',flush=True)
        y=np.tile([-1,1],8);truth=np.tile([-1,1],4);report['classification']={}
        for stage in stages:
            X=np.array([r[stage] for r in records['train']])
            mean=X.mean(0);scale=X.std(0);scale=np.maximum(scale,1e-4)
            z=(X-mean)/scale;weights=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y)
            result={}
            for condition in ['test','shuffled','input_off']:
                v=np.array([r[stage] for r in records[condition]])
                scores=((v-mean)/scale)@weights
                result[condition]=dict(accuracy=float(np.mean(np.where(scores>=0,1,-1)==truth)),scores=scores.tolist())
            report['classification'][stage]=result
            np.savez_compressed(out/(stage+'-readout.npz'),weights=weights,mean=mean,scale=scale,indices=stages[stage])
        report['readout']='Three temporal voltage bins; ridge penalty 1; training-only scale floor 1e-4; 8 test clips. Exploratory, no biological tuning claim.'
    except Exception as e:
        report['error']=str(e);raise
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)
    print(json.dumps(dict(flash=report['flash_response'],classification=report['classification']),indent=2))

if __name__=='__main__':main()
