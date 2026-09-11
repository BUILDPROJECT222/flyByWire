# Backend decision: native WebGPU/Metal for the full graph

Selected 10 September 2026 after a matched local benchmark on the Apple M3 / 8 GB Mac.

**Use WebGPU through wgpu-native/Metal in the local simulation worker, with a browser interface. Keep NumPy/SciPy for the small existing circuit, reference calculations and current readout training.** Use plain WGSL kernels and explicit buffers; a general-purpose model runtime is unnecessary for this sparse operation. Browser-native execution can share the shader later, but it is not the primary runtime for drone-connected experiments.

This selects an execution architecture. It does not select LIF over graded-rate neuroscience, claim a trained full brain, or activate physical drone control. The compact experiment remains on CPU. A subsequent full spiking mode now runs on WebGPU/Metal; see [its model card](../FULL_BRAIN.md). The rate benchmark below remains a separate workload and selection policy.

## Measured comparison

Graph: all 165,122 annotation records with `status=Traced`, retaining 25,563,197 directed edges and 124,025,046 contacts between them. Glia, fragments and other statuses are excluded. This differs deliberately from Neural Canvas's 166,700-entry retention policy. Contact counts are converted into approximately signed, input-normalized float32 weights. The same synthetic drive, initial state and leaky-rate update are used for every backend.

Median **milliseconds per integration step**:

| Workload | SciPy CPU | WebGPU, one thread per neuron | WebGPU, 64 threads per neuron |
|---|---:|---:|---:|
| 1,252-neuron visual circuit | 0.0173 | 0.6989 | 0.6897 |
| 165,122-neuron traced graph | 21.8373 | 7.0429 | **4.7657** |
| Traced graph, 100-step GPU batches | — | 6.3345 | **3.9480** |

The first two GPU rows submit two steps and read the entire state back; the displayed number is elapsed time divided by two. Thirty samples are taken. The final row uses five 100-step batches with one full readback each. CPU time includes arithmetic; GPU time includes command encoding/submission, completion and readback. This is conservative for resident GPU use, but still not a browser or complete application measurement.

The cooperative GPU kernel is about **4.6× faster** than the CPU baseline for short full-graph batches, or **5.5×** when batching 100 steps. For the tiny circuit, dispatch/readback dominates and CPU wins. Cooperative workgroup reduction addresses uneven incoming connection counts better than serial summation within one GPU thread in this test.

The two GPU implementations matched the CPU trajectory after 100 steps: maximum absolute rate error was 1.49e-7 for the scalar kernel and 7.75e-7 for the cooperative kernel on the traced graph, below the declared 2e-5 tolerance. This checks one deterministic synthetic stimulus and seed, not every possible neural state or long-run trajectory.

The complete clean benchmark took 11.35 seconds. macOS `time -l` reported maximum resident set size 689,618,944 bytes and peak memory footprint 980,223,152 bytes. These are process resource measurements for the benchmark, not a measurement of future browser or training memory. The CSR graph itself occupies 205,166,068 bytes. See [raw results](results.json), [timed run](run.log), and [resource report](resources.log).

## Options and decision

| Option | Advantages | Main cost/limit | Choice |
|---|---|---|---|
| Native WebGPU via wgpu-py/wgpu-native → Metal | Measured acceleration; portable WGSL; integrates with Python, RTSP, logs and UDP bridge; simulation lifetime independent of a browser tab | No automatic differentiation; browser execution needs separate validation | **Primary full-graph inference backend** |
| Browser WebGPU in a dedicated Worker | Local installation-free computation; GPU visualization can share state; useful standalone simulator | Tab lifecycle/device loss, browser adapter limits, and camera/UDP bridge remain; unbenchmarked here | Secondary portable viewer/simulator target |
| NumPy/SciPy CSR CPU | Simplest reproducible reference; fastest for our small circuit; existing training works | Full graph is slower in measured comparison | **Keep as small-model backend and oracle** |
| Native C++ sparse/event-driven engine | Can exploit sparse spikes and optimize CPU execution | Additional compiled kernel maintenance; not measured in this comparison; different LIF dynamics cannot be timed as equivalent to our rate model | Alternate for a later matched spiking benchmark |
| PyTorch/MPS or MLX | Useful tools for differentiable training on supported operations | Sparse support, recurrent memory and exact model portability need independent tests | Separate training investigation, not current inference choice |
| CUDA-based simulators | Existing large-network research implementations and training tools | Require different GPU hardware; not a native M3 option | Future large-training workstation path |

This is not a claim that WebGPU beats every optimized CPU or Metal implementation. Only the SciPy and two WGSL implementations were measured. Native Metal-only code might improve performance, but would sacrifice portability without evidence that the additional work is needed.

[wgpu](https://github.com/gfx-rs/wgpu) provides the cross-platform implementation; [wgpu-py](https://wgpu-py.readthedocs.io/en/stable/guide.html) exposes it to Python. The benchmark used version 0.32.0 and confirmed the Apple M3 Metal adapter. Browser overhead, browser GPU availability, battery/thermal behavior under prolonged use and p99 application latency remain unmeasured.

## Why plain WGSL rather than adopting an entire demo

Neural Canvas provided a useful reference for resident buffers and sparse propagation. Its kernel wrapper and crafted body controller are not required for this lab's rate-model operation. Our benchmark shader is an independent implementation of the explicitly stated rate equation, not a port of its spiking model. The selected 64-lane incoming-edge reduction should not be assumed optimal for event-driven LIF: that requires its own matched benchmark with refractory periods, delays, spike density and firing-pattern comparisons.

Retain separate decisions for **runtime**, **neural equations**, **learning rule**, and **sensory/action mappings**. Selecting a fast runtime does not resolve the latter three.

## Integration contract

1. Keep CSR weights, neuronal state and intermediate updates GPU-resident. Upload sensory drive at each observation; gather compact readouts rather than downloading all neuron rates for every integration step.
2. Publish decimated activity for visualization and full snapshots only on demand. Preserve capture, neural and wall clocks independently.
3. Use the local worker for checkpoints, experiment scheduling, camera decode and any later separately armed hardware bridge. Closing/reloading the UI should not own the simulation or actuator lifecycle.
4. Preserve the CPU model as a selectable reference and use it in parity tests. GPU device loss must be explicit; do not silently change model or clock to maintain an apparent frame rate.
5. Keep current output-layer training on CPU while small. Future gradient-based full-graph learning needs a separate memory/throughput study; WebGPU alone does not supply autograd.
6. Before integration, implement input-driven parity tests across multiple seeds, inhibition/excitation, reset/checkpoint restore, boundary cases and sustained workload. The benchmark is sufficient for choosing a direction, not for certifying hardware control.

## Reproduce

From the project root, using its environment:

```sh
.venv/bin/python -m pip install -r benchmarks/requirements.txt
.venv/bin/python benchmarks/prepare_graph.py
/usr/bin/time -l .venv/bin/python benchmarks/compare_backends.py
```

The GPU process needs normal macOS graphics access; the agent filesystem sandbox hid Metal adapters until the GPU run was allowed outside that sandbox. Preparation uses the already downloaded official data. Run preparation to completion before timing. The final saved run was sequential, with no concurrent data preparation. A preliminary run that overlapped preparation was superseded.

Files: `prepare_graph.py` creates the traced graph and manifest; `rate.wgsl` contains both kernels; `compare_backends.py` benchmarks and checks parity; `results.json` records graph/shader hashes and metrics. No drone commands are sent.
