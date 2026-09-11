"""Frozen quadratic decoder for 12 frame-boundary T4/T5 activity samples.
Direction scores are uncalibrated. Workspace may expose them for a clamped yaw-assist bias; the UDP control process still owns motors and never loads the connectome.
"""
from collections import deque
import numpy as np

class MotionReadout:
    def __init__(self,path):
        with np.load(path) as f:
            self.train=f['train'].astype(np.float64);self.mean=f['mean'];self.scale=f['scale']
            self.dual=f['dual'];self.indices=f['indices']
        self.history=deque(maxlen=12)
    def reset(self):self.history.clear()
    def score_features(self,features):
        a=np.asarray(features)
        if a.shape!=self.mean.shape or not np.all(np.isfinite(a)):
            raise ValueError('Invalid temporal feature vector')
        z=(a-self.mean)/self.scale
        return float((1+z.astype(np.float64)@self.train.T/len(z))**2@self.dual)
    def update(self,full_state):
        state=np.asarray(full_state)
        if state.ndim!=1 or len(state)<=int(self.indices.max()) or not np.all(np.isfinite(state)):
            raise ValueError('Invalid full-network state')
        self.history.append(state[self.indices].copy())
        if len(self.history)<12:return None
        features=np.concatenate([part.mean(0) for part in np.array_split(np.array(self.history),3)])
        score=self.score_features(features)
        return dict(direction='right' if score>=0 else 'left',score=score,
                    calibrated_confidence=False,frames=12)

