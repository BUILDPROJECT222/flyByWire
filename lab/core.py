"""Declared leaky-rate approximation over a measured MaleCNS visual subgraph."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'malecns'
MODEL_VERSION = 'visual-rate-v2-eight-bins'
GROUPS = ['retina', 'lamina', 'medulla', 'motion']

class Circuit:
    def __init__(self, condition='intact'):
        self.meta = json.loads((DATA/'visual-circuit.json').read_text())
        self.nodes = self.meta['nodes']
        self.n = len(self.nodes)
        graph = np.load(DATA/'visual-circuit.npz')
        self.source, self.target, self.contacts = graph['source'], graph['target'], graph['weight']
        sign = np.array([-1 if n['inhibitory'] else 1 for n in self.nodes],dtype=np.float32)
        weights = self.contacts * sign[self.source]
        norm = np.bincount(self.target, weights=self.contacts, minlength=self.n)
        weights /= np.maximum(norm[self.target], 1)
        if condition == 'shuffled':
            # Permutes signed weights over fixed edges: degree/topology preserved.
            weights = np.random.default_rng(2026).permutation(weights)
        if condition == 'disconnected': weights = weights * 0
        self.W = csr_matrix((weights,(self.target,self.source)),shape=(self.n,self.n))
        self.retina = np.array([i for i,n in enumerate(self.nodes) if n['group']=='retina'])
        hexes = np.array([n['hex'] for n in self.nodes],dtype=float)
        xmin,xmax = hexes[:,0].min(),hexes[:,0].max()
        self.bins = np.clip(((hexes[:,0]-xmin)/(xmax-xmin+1e-6)*8).astype(int),0,7)
        self.types = sorted(set(n['type'] for n in self.nodes if n['group']!='retina'))
        self.pools = [np.array([i for i,n in enumerate(self.nodes) if n['type']==typ and self.bins[i]//2==bin]) for typ in self.types for bin in range(4)]
        self.feature_size = len(self.pools)+1
        self.group_indices = {g:np.array([i for i,n in enumerate(self.nodes) if n['group']==g]) for g in GROUPS}
        self.rates = np.zeros(self.n,dtype=np.float32)
        self.previous = np.zeros(self.n,dtype=np.float32)

    def reset(self):
        self.rates.fill(0); self.previous.fill(0)

    def step(self, luminance):
        self.previous[:] = self.rates
        drive = np.full(self.n,0.12,dtype=np.float32) # declared tonic drive
        drive[self.retina] += np.asarray(luminance,dtype=np.float32)[self.bins[self.retina]] * 1.5
        for _ in range(2):
            target = np.maximum(0,np.tanh(1.15*self.W.dot(self.rates) + drive))
            self.rates += .35*(target-self.rates)
        # Centered pooled visual activity; no direct observation-to-action bypass.
        features = np.array([float(np.mean(self.rates[p])) if len(p) else 0 for p in self.pools]+[1.0])
        features[:-1] = (features[:-1]-.2)*3
        return features

    def activity(self):
        return {g:float(np.mean(self.rates[i])) if len(i) else 0 for g,i in self.group_indices.items()}

class Arena:
    """Planar kinematic exploration benchmark, not a quadrotor physics simulator."""
    dt=.1
    def __init__(self,seed=11):
        self.seed=seed; rng=np.random.default_rng(seed)
        self.walls=[[0,0,12,0],[12,0,12,12],[12,12,0,12],[0,12,0,0]]
        offset=float(rng.uniform(-.5,.5))
        self.walls += [[3+offset,2,3+offset,7],[6,5,10,5],[7,8,7,11]]
        self.x=1.5; self.y=1.5; self.heading=float(rng.uniform(-.5,.5))
        self.speed=0.; self.t=0.; self.hits=0; self.distance=0.; self.visited=set(); self.path=[]; self.reward=0.
        self.distances=np.zeros(32); self.light=np.zeros(8)
        self.observe()
    def observe(self):
        angles=self.heading+np.linspace(-1.3,1.3,32)
        dx,dy=np.cos(angles),np.sin(angles)
        dist=np.full(32,15.)
        for x1,y1,x2,y2 in self.walls:
            if x1==x2:
                t=(x1-self.x)/np.where(np.abs(dx)>1e-8,dx,1e-8); cross=self.y+t*dy
                valid=(t>0)&(cross>=min(y1,y2))&(cross<=max(y1,y2))
            else:
                t=(y1-self.y)/np.where(np.abs(dy)>1e-8,dy,1e-8); cross=self.x+t*dx
                valid=(t>0)&(cross>=min(x1,x2))&(cross<=max(x1,x2))
            dist=np.minimum(dist,np.where(valid,t,15.))
        self.distances=dist
        # Engineered visual stimulus: brighter wall pixels indicate proximity.
        self.light=(1/(1+dist)).reshape(8,4).mean(axis=1)
        return self.light
    def step(self,action):
        turn,forward=map(float,action)
        self.heading += float(np.clip(turn,-1,1))*1.7*self.dt
        self.speed += .3*(float(np.clip(forward,0,1))*1.6-self.speed)
        nx=self.x+math.cos(self.heading)*self.speed*self.dt
        ny=self.y+math.sin(self.heading)*self.speed*self.dt
        collision=False
        for a,b,c,d in self.walls:
            vx,vy=c-a,d-b; u=np.clip(((nx-a)*vx+(ny-b)*vy)/(vx*vx+vy*vy),0,1)
            if math.hypot(nx-(a+u*vx),ny-(b+u*vy))<.18: collision=True; break
        if collision: self.hits+=1; self.speed=0.
        else:
            self.distance+=math.hypot(nx-self.x,ny-self.y); self.x,self.y=nx,ny
        cell=(int(self.x*2),int(self.y*2)); new=cell not in self.visited
        self.visited.add(cell)
        self.reward += (1. if new else -.015)-(.7 if collision else 0)-.003*abs(turn)
        self.t+=self.dt; self.path.append([round(self.x,3),round(self.y,3)])
        self.path=self.path[-800:]
        return self.observe()
    def state(self):
        return dict(seed=self.seed,x=self.x,y=self.y,heading=self.heading,speed=self.speed,time=self.t,
                    collisions=self.hits,coverage=len(self.visited),distance=self.distance,reward=self.reward,
                    walls=self.walls,path=self.path,distances=self.distances.tolist(),luminance=self.light.tolist())

def initial_weights(size,seed=7):
    rng=np.random.default_rng(seed)
    w=rng.normal(0,.15,(2,size)); w[1,-1]=.6
    return w

def decode(features,weights):
    raw=np.tanh(weights@features)
    return np.array([raw[0],(raw[1]+1)*.5])

def evaluate(circuit,weights,seeds=(101,102,103),steps=240,baseline=False):
    results=[]
    for seed in seeds:
        env=Arena(seed); circuit.reset()
        for _ in range(steps):
            features=circuit.step(env.light)
            if baseline:
                left,right=np.mean(env.distances[:12]),np.mean(env.distances[-12:])
                front=float(np.min(env.distances[12:20]))
                action=np.array([np.clip((right-left)*.8,-1,1) if front<2 else .08,.8 if front>1 else .1])
            else: action=decode(features,weights)
            env.step(action)
        results.append(dict(seed=seed,reward=env.reward,coverage=len(env.visited),collisions=env.hits))
    return dict(reward=float(np.mean([r['reward'] for r in results])),
                coverage=float(np.mean([r['coverage'] for r in results])),
                collisions=float(np.mean([r['collisions'] for r in results])),episodes=results)
