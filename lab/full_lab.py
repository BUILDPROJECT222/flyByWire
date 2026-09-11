"""Full-brain experiment worker; no motor outputs. GPU commands stay on its thread."""
import json,threading,time
from queue import Queue,Empty
import numpy as np
from .core import DATA

class FullLab:
    def __init__(self):
        self.lock=threading.Lock();self.commands=Queue();self.stop=threading.Event();self.engine=None;self.nodes=[]
        self.frame=dict(status='loading full graph',ready=False,running=False,spikes=[],trace=[],model_ms=0,frame=0,error=None)
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def state(self):
        with self.lock:return dict(self.frame)
    def send(self,command):self.commands.put(command)
    def loop(self):
        try:
            from .full_engine import FullEngine
            meta=json.loads((DATA/'full-brain.json').read_text());self.nodes=meta['nodes'];manifest=meta['manifest']
            graph=np.load(DATA/'full-spiking.npz');self.engine=FullEngine(graph)
            types=np.array([n['type'] for n in self.nodes]);groups=np.array([n['group'] for n in self.nodes])
            presets={'LC4':np.flatnonzero(types=='LC4'),'LC9':np.flatnonzero(types=='LC9'),'DNa02':np.flatnonzero(types=='DNa02'),'retina':np.flatnonzero(types=='R1-R6'),'off':np.array([],int)}
            hz=120.;preset='LC4';selection=presets[preset];running=True;recurrent=True;trace=[];sequence=0
            with self.lock:self.frame.update(ready=True,status='running',manifest=manifest,adapter=dict(self.engine.adapter.info),presets={k:len(v) for k,v in presets.items()})
            while not self.stop.is_set():
                single=False
                while True:
                    try:c=self.commands.get_nowait()
                    except Empty:break
                    if c['action']=='run':running=True
                    elif c['action']=='pause':running=False
                    elif c['action']=='step':running=False;single=True
                    elif c['action']=='reset':
                        self.engine.reset();trace=[]
                        with self.lock:self.frame.update(spikes=[],trace=[],model_ms=0,active=0,total_spikes=0,group_spikes={})
                    elif c['action']=='stimulus':
                        preset=c['preset'];selection=presets[preset];hz=c['hz']
                    elif c['action']=='neuron':preset='selected neuron';selection=np.array([c['index']]);hz=c['hz']
                    elif c['action']=='recurrence':recurrent=c['enabled']
                if running or single:
                    rates=np.zeros(len(self.nodes),np.float32);rates[selection]=hz
                    start=time.perf_counter();counts=self.engine.batch(rates,recurrent=recurrent);elapsed=time.perf_counter()-start
                    active=np.flatnonzero(counts);sequence+=1
                    spikes=np.column_stack([active,counts[active].astype(np.int32)]).tolist()
                    trace=(trace+[dict(time=self.engine.tick*.1,total=int(counts.sum()),active=len(active))])[-120:]
                    with self.lock:self.frame.update(spikes=spikes,trace=trace,model_ms=self.engine.tick*.1,frame=sequence,batch_ms=elapsed*1000,realtime_ratio=.01/elapsed,active=len(active),total_spikes=int(counts.sum()),group_spikes={g:int(counts[groups==g].sum()) for g in manifest['classes']})
                with self.lock:self.frame.update(running=running,status='running' if running else 'paused',stimulus=preset,stimulus_hz=hz,stimulated=len(selection),recurrence=recurrent)
                self.stop.wait(.025 if running else .1)
        except Exception as e:
            with self.lock:self.frame.update(error=str(e),status='GPU worker stopped',running=False)
        finally:
            if self.engine:self.engine.close()
    def node(self,index):
        if not self.nodes or not 0<=index<len(self.nodes):raise ValueError('Neuron index out of range')
        return dict(index=index,**self.nodes[index])
