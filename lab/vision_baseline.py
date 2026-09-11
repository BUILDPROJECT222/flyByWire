"""Image-only translation control for the synthetic grating pilot."""
import json
import numpy as np
from .vision_experiment import stimulus

def direction(frames):
    # Exclude wraparound boundary; compare one-pixel candidate translations.
    previous=frames[:-1];following=frames[1:, :, 1:-1]
    errors=[float(np.mean((following-previous[:,:,2:])**2)),
            float(np.mean((following-previous[:,:,:-2])**2))]
    return -1 if errors[0]<errors[1] else 1

def main():
    outcomes=[]
    for seed in range(100,104):
        for label in [-1,1]:
            pred=direction(stimulus(seed,label))
            outcomes.append(dict(seed=seed,label=label,prediction=pred))
    print(json.dumps(dict(scope='Engineered pixel-translation diagnostic, not matched neural capacity',
        accuracy=sum(x['label']==x['prediction'] for x in outcomes)/len(outcomes),outcomes=outcomes),indent=2))

if __name__=='__main__':main()
