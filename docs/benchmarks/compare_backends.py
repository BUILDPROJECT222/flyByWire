"""Matched CPU/WebGPU benchmark. Native Metal timing is NOT browser timing."""
import json,time,platform,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import wgpu
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lab.core import Circuit

def stats(values):
    return dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples=len(values))

def benchmark(matrix,device,label):
    n=matrix.shape[0];initial=np.random.default_rng(7).uniform(0,.4,n).astype(np.float32)
    drive=np.full(n,.12,np.float32);drive[::31]+=.5
    def cpu(x):return x+np.float32(.35)*(np.maximum(0,np.tanh(np.float32(1.15)*matrix.dot(x)+drive))-x)
    x=initial.copy()
    for _ in range(5):x=cpu(x)
    timings=[]
    for _ in range(30):
        start=time.perf_counter();x=cpu(x);timings.append((time.perf_counter()-start)*1000)
    result=dict(graph=label,nodes=n,edges=int(matrix.nnz),cpu=stats(timings),gpu={})
    expected=initial.copy()
    for _ in range(100):expected=cpu(expected)
    shader=device.create_shader_module(code=(root/'benchmarks/rate.wgsl').read_text())
    usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST|wgpu.BufferUsage.COPY_SRC
    buffers=[device.create_buffer_with_data(data=np.ascontiguousarray(a),usage=usage) for a in [matrix.indptr.astype(np.uint32),matrix.indices.astype(np.uint32),matrix.data.astype(np.float32),initial,np.zeros(n,np.float32)]]
    for kernel in ['scalar','cooperative']:
        start=time.perf_counter()
        pipeline=device.create_compute_pipeline(layout='auto',compute={'module':shader,'entry_point':kernel})
        groups=[]
        for swap in [False,True]:
            ordered=buffers[:3]+([buffers[4],buffers[3]] if swap else buffers[3:])
            groups.append(device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':b}} for i,b in enumerate(ordered)]))
        compilation_ms=(time.perf_counter()-start)*1000
        def run(steps):
            encoder=device.create_command_encoder()
            for i in range(steps):
                p=encoder.begin_compute_pass();p.set_pipeline(pipeline);p.set_bind_group(0,groups[i%2])
                if kernel=='scalar':p.dispatch_workgroups((n+63)//64)
                else:p.dispatch_workgroups(min(4096,n),(n+4095)//4096)
                p.end()
            device.queue.submit([encoder.finish()])
            # Full-state readback enforces completion; included in timings.
            return np.frombuffer(device.queue.read_buffer(buffers[3 if steps%2==0 else 4]),np.float32).copy()
        device.queue.write_buffer(buffers[3],0,initial)
        actual=run(100)
        error=float(np.max(np.abs(actual-expected)))
        if not np.isfinite(actual).all() or error>2e-5:raise RuntimeError(f'{label}/{kernel}: numerical mismatch {error}')
        times=[]
        for _ in range(30):
            start=time.perf_counter();run(2);times.append((time.perf_counter()-start)*1000/2)
        batches=[]
        for _ in range(5):
            start=time.perf_counter();run(100);batches.append((time.perf_counter()-start)*1000/100)
        result['gpu'][kernel]=dict(two_step_full_readback=stats(times),hundred_step_full_readback=stats(batches),max_abs_error_100_steps=error,compile_ms=compilation_ms)
        print(label,kernel,result['gpu'][kernel],flush=True)
    for b in buffers:b.destroy()
    return result

adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
device=adapter.request_device_sync(required_limits={'max-storage-buffers-per-shader-stage':5})
print(dict(adapter.info),flush=True)
with (root/'data/malecns/benchmark-full-csr.npz').open('rb') as stream: graph_hash=hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(date='2026-09-10',graph_sha256=graph_hash,validation_steps=100,initial_state_seed=7,drive='0.12 tonic + 0.5 on each 31st neuron',shader_sha256=hashlib.sha256((root/'benchmarks/rate.wgsl').read_bytes()).hexdigest(),graph_manifest=json.loads((root/'data/malecns/benchmark-full-csr.json').read_text()),platform=platform.platform(),wgpu=wgpu.__version__,adapter=dict(adapter.info),
 scope='Native wgpu/Metal, not browser. Synthetic fixed drive; matched normalized signed rate dynamics, not LIF or biological validation. CPU includes arithmetic; GPU includes dispatch, completion and full-state readback. GPU results are per integration step.',results=[])
for label,matrix in [('visual-1252',Circuit().W),('full-malecns',load_npz(root/'data/malecns/benchmark-full-csr.npz'))]:
 report['results'].append(benchmark(matrix,device,label))
(root/'benchmarks/results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
