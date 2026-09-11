"""Cross-entropy search of an artificial readout. Circuit weights stay fixed."""
import json
import time
from pathlib import Path
import numpy as np
from .core import Circuit,evaluate,initial_weights,ROOT,MODEL_VERSION

def train(generations=8,seed=7,callback=lambda _:None,cancel=lambda:False):
    circuit=Circuit(); rng=np.random.default_rng(seed)
    mean=initial_weights(circuit.feature_size,seed); std=np.full_like(mean,.45)
    best=mean.copy(); best_score=-float('inf'); history=[]
    started=time.time()
    # Fixed training seeds; disjoint evaluation seeds are never used for selection.
    for generation in range(generations):
        if cancel(): break
        candidates=mean+rng.normal(size=(12,*mean.shape))*std
        candidates[0]=best
        scores=[]
        for weights in candidates:
            if cancel(): break
            scores.append(evaluate(circuit,weights,seeds=(11,23),steps=160)['reward'])
        if len(scores)!=len(candidates): break
        elite=np.argsort(scores)[-3:]
        mean=.3*mean+.7*np.mean(candidates[elite],axis=0)
        std=np.maximum(.06,.4*std+.6*np.std(candidates[elite],axis=0))
        win=int(np.argmax(scores))
        if scores[win]>best_score: best=candidates[win].copy(); best_score=float(scores[win])
        item=dict(generation=generation+1,reward=best_score,mean_reward=float(np.mean(scores)),
                  elapsed_s=time.time()-started)
        history.append(item); callback(item)
    path=ROOT/'experiments'/time.strftime('%Y%m%dT%H%M%S')
    path.mkdir(parents=True,exist_ok=True)
    np.save(path/'readout.npy',best)
    evaluation=evaluate(circuit,best)
    baseline=evaluate(circuit,initial_weights(circuit.feature_size,seed))
    result=dict(run_id=path.name,model_version=MODEL_VERSION,method='Cross-entropy optimization of artificial readout only',
                seed=seed,population=12,elite=3,train_seeds=[11,23],evaluation_seeds=[101,102,103],
                training_episode_steps=160,evaluation_episode_steps=240,generations=len(history),
                cancelled=cancel(),history=history,evaluation=evaluation,untrained=baseline,
                elapsed_s=time.time()-started,scope='Planar simulated exploration; not physical drone learning',
                graph_manifest=circuit.meta['manifest'])
    (path/'report.json').write_text(json.dumps(result,indent=2))
    return best,result
