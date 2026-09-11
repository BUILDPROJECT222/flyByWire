"""Offline full-graph visual pilot, with held-out phases and fixed-readout controls."""
import argparse,hashlib,json,subprocess,time
from datetime import datetime,timezone
import numpy as np
from .core import DATA,ROOT
from .full_engine import FullEngine
from .full_vision import RetinaEncoder
from .validate_full import sha


def stimulus(seed,direction,frames=12):
    rng=np.random.default_rng(seed)
    phase=rng.uniform(0,2*np.pi)
    x=np.linspace(0,1,64)[None,:]
    y=np.ones((48,1))
    return np.array([127.5+100*np.sin(2*np.pi*x*3+phase-direction*t*.35)*y
                     for t in range(frames)],np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lamina-background-hz',type=float,default=0)
    args=parser.parse_args()
    if not 0<=args.lamina_background_hz<=120:parser.error('Background must be 0..120 Hz')
    out=ROOT/'experiments'/('full-vision-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    meta=json.loads((DATA/'full-brain.json').read_text())
    lamina=np.array([i for i,n in enumerate(meta['nodes']) if n['type'] in ['L1','L2','L3']])
    output=np.array([i for i,n in enumerate(meta['nodes']) if n['group']=='descending_neuron'])
    encoder=RetinaEncoder()
    engine=FullEngine(np.load(DATA/'full-spiking.npz'))
    report=dict(scope='Small offline sensory pilot; no motor output or biological direction-selectivity claim.',
        lamina_background_hz=args.lamina_background_hz,
        background_note='Engineered constant Poisson drive to L1/L2/L3, not fitted physiology.',
        train_seeds=list(range(8)),test_seeds=list(range(100,104)),
        frames_per_clip=12,neural_ms_per_frame=20,
        output_indices=output.tolist(),adapter=dict(engine.adapter.info),
        files={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'full-retina.json',DATA/'full-spiking.npz',
            ROOT/'lab/full_engine.py',ROOT/'lab/shaders/full-spiking.wgsl',ROOT/'lab/full_vision.py',
            ROOT/'lab/vision_experiment.py']},trials=[])
    def run(frames,name,noise,recurrence=True):
        engine.reset()
        pooled=[];rows=[];times=[];window=0
        for frame in frames:
            rates=encoder.encode(frame)
            rates[lamina]=args.lamina_background_hz
            frame_counts=np.zeros(len(output),np.float32)
            for _ in range(2):
                t=time.perf_counter()
                counts=engine.batch(rates,recurrent=recurrence,seed=noise)
                times.append((time.perf_counter()-t)*1000)
                if not np.all(np.isfinite(counts)):raise RuntimeError('Nonfinite counts')
                frame_counts+=counts[output]
                ids=np.flatnonzero(counts)
                rows.append(np.column_stack([np.full(len(ids),window),ids,counts[ids]]).astype(np.uint32))
                window+=1
            pooled.append(frame_counts)
        pooled=np.array(pooled)
        np.savez_compressed(out/(name+'.npz'),spikes=np.concatenate(rows),output_counts=pooled,
                            input_frames=frames,wall_ms=times,output_indices=output)
        # Three fixed temporal bins, no direct sensory shortcut into the readout.
        features=np.concatenate([part.mean(0) for part in np.array_split(pooled,3)])
        report['trials'].append(dict(name=name,noise_seed=noise,recurrence=recurrence,
            neural_ms=len(frames)*20,total_output_spikes=int(pooled.sum()),
            batch_p95_ms=float(np.percentile(times,95))))
        return features
    try:
        X=[];labels=[];test=[];testlabels=[];controls={'shuffled':[],'disconnected':[]}
        for split,seeds in [('train',report['train_seeds']),('test',report['test_seeds'])]:
            for seed in seeds:
                for direction in [-1,1]:
                    frames=stimulus(seed,direction)
                    name=f'{split}-{seed}-{direction}'
                    features=run(frames,name,seed+1000)
                    (X if split=='train' else test).append(features)
                    (labels if split=='train' else testlabels).append(direction)
                    if split=='test':
                        order=np.random.default_rng(seed+2000).permutation(len(frames))
                        controls['shuffled'].append(run(frames[order],name+'-shuffled',seed+1000))
                        controls['disconnected'].append(run(frames,name+'-disconnected',seed+1000,False))
                print(split,seed,'finished',flush=True)
        X=np.array(X);test=np.array(test);y=np.array(labels);truth=np.array(testlabels)
        mean=X.mean(0);scale=X.std(0);scale[scale<1]=1
        z=(X-mean)/scale
        # Fixed ridge penalty, fit in sample space. No tuning on test phases.
        weights=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y.astype(float))
        def score(a):
            value=((np.array(a)-mean)/scale)@weights
            predictions=np.where(value>=0,1,-1)
            return dict(accuracy=float(np.mean(predictions==truth)),predictions=predictions.tolist(),
                        scores=value.tolist(),labels=truth.tolist())
        report['classification']={name:score(a) for name,a in dict(intact=test,**controls).items()}
        report['classification']['chance']=.5
        report['classification']['test_clips']=len(truth)
        report['training_accuracy']=float(np.mean(np.where(z@weights>=0,1,-1)==y))
        report['pilot_gate']=dict(
            intact_above_chance=report['classification']['intact']['accuracy']>.5,
            degraded_by_temporal_shuffle=report['classification']['intact']['accuracy']>report['classification']['shuffled']['accuracy'],
            degraded_by_disconnection=report['classification']['intact']['accuracy']>report['classification']['disconnected']['accuracy'])
        np.savez_compressed(out/'readout.npz',weights=weights,mean=mean,scale=scale,
                            train_features=X,test_features=test,train_labels=y,test_labels=truth)
        recording=ROOT/'recordings/20260909T192201Z/takeoff-land.mp4'
        # The saved MP4 is already rotated clockwise; no second transpose.
        decode=subprocess.run(['ffmpeg','-v','error','-i',str(recording),'-vf','fps=2,scale=64:48',
            '-frames:v','12','-pix_fmt','gray','-f','rawvideo','pipe:1'],capture_output=True,check=True,timeout=30)
        frames=np.frombuffer(decode.stdout,np.uint8).reshape(-1,48,64)
        if len(frames)!=12:raise RuntimeError('Expected 12 replay frames')
        run(frames,'drone-replay',9001)
        report['replay']=dict(path=str(recording.relative_to(ROOT)),sha256=sha(recording),
            frames=12,sampling_fps=2,neural_ms=240,
            note='First 12 frames sampled at 2 Hz; each held for 20 ms neural time. Offline compressed-time response, not real-time flight inference. Unlabeled; not included in direction accuracy.')
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        engine.close()
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)
    print(json.dumps(report['classification'],indent=2))
    print('Pilot gates:',report['pilot_gate'])


if __name__=='__main__':main()
