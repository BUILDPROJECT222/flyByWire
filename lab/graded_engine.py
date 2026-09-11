"""Experimental graded state on the complete retained graph; not a spiking engine."""
import json
from pathlib import Path
import numpy as np
import wgpu
from scipy.sparse import csr_matrix
from .core import DATA

def prepare():
    with np.load(DATA/'full-spiking.npz') as f:g={k:f[k] for k in f.files}
    n=len(g['sign'])
    outgoing=csr_matrix((g['counts'].astype(np.float32),g['targets'],g['offsets']),shape=(n,n))
    incoming=outgoing.T.tocsr()
    denominator=np.asarray(incoming.sum(1)).ravel();denominator[denominator==0]=1
    incoming.data*=g['sign'][incoming.indices]
    incoming.data/=np.repeat(denominator,np.diff(incoming.indptr))
    # Keep zero-sign entries so retained structure matches the original graph.
    np.savez(DATA/'full-graded.npz',offsets=incoming.indptr.astype(np.uint32),
             sources=incoming.indices.astype(np.uint32),weights=incoming.data.astype(np.float32))
    print('Prepared graded graph:',n,'neurons',incoming.nnz,'edges')

class GradedEngine:
    dt_ms=5
    def __init__(self,graph,bias):
        self.n=len(bias);self.bias=np.asarray(bias,np.float32);self.tick=0;self.current=0
        self.adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        self.device=self.adapter.request_device_sync();d=self.device;u=wgpu.BufferUsage
        def buffer(data):
            return d.create_buffer_with_data(data=np.asarray(data),usage=u.STORAGE|u.COPY_SRC|u.COPY_DST)
        self.structure=[buffer(graph[k]) for k in ['offsets','sources','weights']]
        self.states=[buffer(self.bias),buffer(self.bias)]
        self.drive=buffer(np.zeros(self.n,np.float32));self.bias_buffer=buffer(self.bias)
        self.alpha=buffer(np.full(self.n,.1,np.float32))
        shader=d.create_shader_module(code=(Path(__file__).parent/'shaders/full-graded.wgsl').read_text())
        self.pipeline=d.create_compute_pipeline(layout='auto',compute=dict(module=shader,entry_point='advance'))
        self.groups=[d.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),
            entries=[dict(binding=i,resource=dict(buffer=b)) for i,b in enumerate(
                self.structure+[self.states[j],self.states[1-j],self.drive,self.bias_buffer,self.alpha])]) for j in [0,1]]
    def configure(self,bias,tau_ms):
        bias=np.asarray(bias,np.float32);tau=np.asarray(tau_ms,np.float32)
        if (bias.shape!=(self.n,) or tau.shape!=(self.n,) or
            not np.all(np.isfinite(bias)) or not np.all(np.isfinite(tau)) or
            np.any(tau<10) or np.any(tau>1000)):
            raise ValueError("Finite per-cell bias and tau in 10..1000 ms required")
        self.bias=bias.copy()
        self.device.queue.write_buffer(self.bias_buffer,0,self.bias)
        self.device.queue.write_buffer(self.alpha,0,np.asarray(5/tau,np.float32))
        self.reset()
    def reset(self,initial=None):
        value=self.bias if initial is None else np.asarray(initial,np.float32)
        if value.shape!=(self.n,) or not np.all(np.isfinite(value)):raise ValueError("Invalid initial state")
        for state in self.states:self.device.queue.write_buffer(state,0,value)
        self.tick=0;self.current=0
    def batch(self,drive,steps=4):
        a=np.asarray(drive,np.float32)
        if a.shape!=(self.n,) or not np.all(np.isfinite(a)) or not 1<=steps<=1000:
            raise ValueError('Invalid drive or step count')
        d=self.device;d.queue.write_buffer(self.drive,0,a);encoder=d.create_command_encoder()
        for _ in range(steps):
            p=encoder.begin_compute_pass();p.set_pipeline(self.pipeline)
            p.set_bind_group(0,self.groups[self.current])
            p.dispatch_workgroups(min(self.n,4096),(self.n+4095)//4096);p.end()
            self.current=1-self.current
        d.queue.submit([encoder.finish()]);self.tick+=steps
        return np.frombuffer(d.queue.read_buffer(self.states[self.current]),np.float32).copy()
    def close(self):
        for b in self.structure+self.states+[self.drive,self.bias_buffer,self.alpha]:b.destroy()
        self.device.destroy()

if __name__=='__main__':prepare()
