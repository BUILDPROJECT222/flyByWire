"""Resident full-graph WebGPU LIF engine, adapted from the MIT Neural Canvas shader."""
import math,struct
from pathlib import Path
import numpy as np
import wgpu

class FullEngine:
    def __init__(self,graph):
        self.offsets=graph['offsets'];self.n=len(self.offsets)-1;self.edges=len(graph['targets']);self.tick=0
        self.adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        self.device=self.adapter.request_device_sync(required_limits={'max-storage-buffers-per-shader-stage':8})
        d=self.device;U=wgpu.BufferUsage
        def buffer(data=None,size=None,extra=0):
            usage=U.STORAGE|U.COPY_SRC|U.COPY_DST|extra
            return d.create_buffer_with_data(data=data,usage=usage) if data is not None else d.create_buffer(size=size,usage=usage)
        packed=np.concatenate([self.offsets,graph['sign'].view(np.uint32),graph['targets']]).astype(np.uint32)
        self.graph=buffer(packed);self.contacts=buffer(graph['counts']);self.state=buffer(size=self.n*16)
        self.history=buffer(size=(19+self.n*19)*4);self.rates=buffer(size=self.n*4)
        self.counts=buffer(size=self.n*4);self.currents=buffer(size=self.n*4)
        self.indirect=buffer(size=19*12,extra=U.INDIRECT)
        self.uniform=d.create_buffer(size=256*100,usage=U.UNIFORM|U.COPY_DST)
        entries=[dict(binding=i,visibility=wgpu.ShaderStage.COMPUTE,buffer=dict(type='uniform',has_dynamic_offset=True,min_binding_size=32) if i==7 else dict(type='read-only-storage' if i in [0,1,4] else 'storage')) for i in range(8)]
        layout=d.create_bind_group_layout(entries=entries)
        ilayout=d.create_bind_group_layout(entries=[dict(binding=0,visibility=wgpu.ShaderStage.COMPUTE,buffer=dict(type='storage'))])
        module=d.create_shader_module(code=(Path(__file__).parent/'shaders/full-spiking.wgsl').read_text())
        self.pipelines=[d.create_compute_pipeline(layout=d.create_pipeline_layout(bind_group_layouts=[layout] if i==0 else [layout,ilayout]),compute=dict(module=module,entry_point=name)) for i,name in enumerate(['propagate','advance'])]
        self.buffers=[self.graph,self.contacts,self.state,self.history,self.rates,self.counts,self.currents,self.uniform]
        self.bind=d.create_bind_group(layout=layout,entries=[dict(binding=i,resource=dict(buffer=b,**({'size':32} if i==7 else {}))) for i,b in enumerate(self.buffers)])
        self.ibind=d.create_bind_group(layout=ilayout,entries=[dict(binding=0,resource=dict(buffer=self.indirect))])
        self.reset()
    def reset(self):
        self.tick=0;state=np.zeros((self.n,4),np.float32);state[:,0]=-52
        d=self.device;d.queue.write_buffer(self.state,0,state)
        indirect=np.ones((19,3),np.uint32);indirect[:,0]=0;d.queue.write_buffer(self.indirect,0,indirect)
        e=d.create_command_encoder()
        for b in [self.history,self.counts,self.currents,self.rates]:e.clear_buffer(b)
        d.queue.submit([e.finish()])
    def batch(self,rates,steps=100,recurrent=True,seed=7):
        if not 1<=steps<=100 or len(rates)!=self.n:raise ValueError('Invalid batch')
        d=self.device;d.queue.write_buffer(self.rates,0,np.asarray(rates,np.float32))
        data=bytearray(256*steps);em=math.exp(-.1/20);es=math.exp(-.1/5);coupling=5/15*(em-es)
        for k in range(steps):struct.pack_into('<5I3f',data,k*256,self.n,self.tick+k,self.edges,int(not recurrent),seed,em,es,coupling)
        d.queue.write_buffer(self.uniform,0,data)
        e=d.create_command_encoder();e.clear_buffer(self.counts)
        # Separate passes avoid binding the indirect buffer as writable storage
        # while it is consumed as dispatch arguments.
        for k in range(steps):
            if recurrent and self.tick+k>=18:
                p=e.begin_compute_pass();p.set_pipeline(self.pipelines[0]);p.set_bind_group(0,self.bind,[k*256]);p.dispatch_workgroups_indirect(self.indirect,((self.tick+k+1)%19)*12);p.end()
            p=e.begin_compute_pass();p.set_pipeline(self.pipelines[1]);p.set_bind_group(0,self.bind,[k*256]);p.set_bind_group(1,self.ibind);p.dispatch_workgroups((self.n+127)//128);p.end()
        d.queue.submit([e.finish()]);counts=np.frombuffer(d.queue.read_buffer(self.counts),np.float32).copy();self.tick+=steps
        return counts
    def snapshot(self):
        return np.frombuffer(self.device.queue.read_buffer(self.state),np.float32).reshape(self.n,4).copy()
    def close(self):
        for b in self.buffers+[self.indirect]:b.destroy()
        self.device.destroy()
