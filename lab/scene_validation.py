"""Frozen full-graph decoder: new scenes, continuous input, and unlabeled drone replay."""
import json,shutil,subprocess,time
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from .core import ROOT,DATA
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .motion_readout import MotionReadout
from .scene_stimuli import clip,image_motion,texture
from .validate_full import sha

CHECKPOINT=ROOT/'experiments/visual-fit-20260910T133833056081Z/quadratic-0-T4_T5.npz'
RECORDING=ROOT/'recordings/20260909T192201Z/takeoff-land.mp4'

def main():
    out=ROOT/'experiments'/('scene-validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    (out/'source').mkdir()
    sources=['lab/scene_validation.py','lab/scene_stimuli.py','lab/graded_engine.py',
             'lab/shaders/full-graded.wgsl','lab/full_vision.py','lab/motion_readout.py']
    for p in sources:shutil.copy2(ROOT/p,out/'source'/p.replace('/','__'))
    report=dict(scope='Frozen sensory generalization and replay test; no aircraft control',
        seeds=list(range(2000,2020)),trials=[],continuous={},recording={},
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [CHECKPOINT,RECORDING,DATA/'full-graded.npz',DATA/'full-retina.json']+[ROOT/p for p in sources]})
    nodes=json.loads((DATA/'full-brain.json').read_text())['nodes'];encoder=RetinaEncoder();readout=MotionReadout(CHECKPOINT)
    with np.load(DATA/'full-graded.npz') as f:graph={k:f[k] for k in f.files}
    engine=GradedEngine(graph,np.array([0 if n['type']=='R1-R6' else .5 for n in nodes],np.float32))
    try:
        baseline=engine.batch(np.zeros(len(nodes),np.float32),200)
        def step(image):
            start=time.perf_counter();v=engine.batch(encoder.encode(image)/120,4);prediction=readout.update(v)
            return prediction,(time.perf_counter()-start)*1000
        def reset():engine.reset(baseline);readout.reset()
        for seed in report['seeds']:
            for kind in ['texture','on_edge','off_edge','stationary','looming']:
                for label in ([-1,1] if kind in ['texture','on_edge','off_edge'] else [0]):
                    frames=clip(seed,kind,label or 1);reset();times=[]
                    for frame in frames:prediction,elapsed=step(frame);times.append(elapsed)
                    predicted=1 if prediction['direction']=='right' else -1
                    report['trials'].append(dict(seed=seed,kind=kind,label=label,prediction=predicted,
                        score=prediction['score'],pixel_baseline=image_motion(frames),
                        p95_processing_ms=float(np.percentile(times,95))))
                    np.savez_compressed(out/f'{kind}-{seed}-{label}.npz',frames=frames,processing_ms=times)
            print('scene seed',seed,'complete',flush=True)
        summary={}
        draws=np.random.default_rng(3210).integers(0,20,(10000,20))
        for kind in ['texture','on_edge','off_edge']:
            trials=[t for t in report['trials'] if t['kind']==kind]
            paired=np.array([np.mean([t['prediction']==t['label'] for t in trials if t['seed']==s]) for s in report['seeds']])
            summary[kind]=dict(correct=sum(t['prediction']==t['label'] for t in trials),clips=len(trials),
                accuracy=float(paired.mean()),phase_bootstrap_95=np.quantile(paired[draws].mean(1),[.025,.975]).tolist(),
                pixel_baseline_accuracy=float(np.mean([t['pixel_baseline']==t['label'] for t in trials])))
        for kind in ['stationary','looming']:
            trials=[t for t in report['trials'] if t['kind']==kind]
            summary[kind]=dict(clips=len(trials),forced_horizontal_outputs=len(trials),
                note='Binary decoder has no stationary/looming/unknown category; these outputs are not validated motion detections.')
        report['summary']=summary
        # Continuous sequence: stationary, right, left, stationary. Never reset between segments.
        canvas=texture(3030);position=64;frames=[];labels=[]
        for i in range(72):
            label=0 if i<12 or i>=60 else 1 if i<36 else -1
            position-=label;frames.append(canvas[8:56,position:position+64]);labels.append(label)
        reset();trace=[]
        for i,frame in enumerate(frames):
            prediction,elapsed=step(frame)
            trace.append(dict(frame=i,source_time_s=i/10,label=labels[i],prediction=prediction,processing_ms=elapsed))
        mature=[x for x in trace if x['label']!=0 and x['frame']>=11 and len(set(labels[x['frame']-11:x['frame']+1]))==1]
        continuous=dict(trace=trace,assumed_source_fps=10,
            stable_motion_windows=len(mature),
            stable_motion_accuracy=float(np.mean([(1 if x['prediction']['direction']=='right' else -1)==x['label'] for x in mature])),
            note='Overlapping windows are dependent; no population confidence interval. Frame times describe input schedule; run is unpaced.')
        for boundary in [12,36,60]:
            segment=trace[boundary:boundary+24 if boundary<60 else 72]
            label=labels[boundary]
            correct=[x['frame'] for x in segment if x['prediction'] and (1 if x['prediction']['direction']=='right' else -1)==label]
            continuous.setdefault('transitions',[]).append(dict(frame=boundary,label=label,
                first_correct_delay_s=(correct[0]-boundary)/10 if correct else None))
        report['continuous']=continuous
        np.savez_compressed(out/'continuous.npz',frames=frames,labels=labels)
        # Decode original already-rotated recording. Resize with PIL, matching the workspace.
        process=subprocess.run(['ffmpeg','-v','error','-i',str(RECORDING),'-vf','fps=10,scale=320:240',
            '-pix_fmt','rgb24','-f','rawvideo','pipe:1'],capture_output=True,check=True,timeout=30)
        frames=np.frombuffer(process.stdout,np.uint8).reshape(-1,240,320,3)
        reset();previous=None;trace=[];resets=[]
        for i,frame in enumerate(frames):
            small=np.asarray(Image.fromarray(frame).resize((64,48)),np.float32)
            change=float(np.mean(np.abs(small-previous))) if previous is not None else 0
            if change>45:reset();resets.append(i)
            prediction,elapsed=step(small)
            pixel=image_motion(np.array([previous,small])) if previous is not None else None
            trace.append(dict(frame=i,source_pts_approx_s=i/10,prediction=prediction,
                              processing_ms=elapsed,mean_pixel_change=change,pixel_reference=pixel))
            previous=small
        times=np.array([t['processing_ms'] for t in trace])
        report['recording']=dict(frames=len(frames),sample_fps=10,resets=resets,trace=trace,
            processing_p50_ms=float(np.percentile(times,50)),processing_p95_ms=float(np.percentile(times,95)),
            processing_p99_ms=float(np.percentile(times,99)),processing_max_ms=float(times.max()),
            note='Unlabeled flight footage: no accuracy claim. Processing starts after decode; source timestamps are approximate resampled positions. No pacing, capture latency or display latency measured here.')
    except Exception as e:report['error']=str(e);raise
    finally:
        engine.close();(out/'report.json').write_text(json.dumps(report,indent=2));print('Report:',out/'report.json',flush=True)
    print(json.dumps(report['summary'],indent=2))
    print('Continuous stable-window accuracy:',report['continuous']['stable_motion_accuracy'])
    print('Recording frames:',report['recording']['frames'],'p95:',report['recording']['processing_p95_ms'])

if __name__=='__main__':main()
