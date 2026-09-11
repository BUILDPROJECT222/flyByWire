# Fly connectomes, embodied learning and drone control: research review

Reviewed 10 September 2026. This is a targeted landscape survey of 32 papers, projects and research resources, plus the five supplied screenshots. It is not an exhaustive systematic review. Depth varies: selected methods and source code were inspected closely; other entries were screened from primary abstracts, documentation or repository READMEs. None of the external simulators was installed or benchmarked in this review. Recommendations and hardware estimates below are our assessment, not measured performance on this Mac.

## Recommendation

Keep the current lab as an experiment interface, but develop two comparable model paths:

1. **Whole-MaleCNS experimental path:** benchmark the existing Neural Canvas WebGPU engine and a compact CPU reference before writing another full-graph engine. Keep measured anatomy, stimulation and rates visible. Add a genuine sensory feedback loop and a separately declared trainable readout. Whole-graph simulation is worth testing on this Mac now; whole-graph gradient training is a different resource problem.
2. **Task-trained vision path:** reproduce a FlyVis pretrained visual-response experiment, then evaluate optical-flow features on our footage. Use a calibrated drone simulator and train a small navigation policy. This is the strongest immediate route toward useful fly-derived perception with a published validation foundation.

Run both against conventional visual and control baselines. A full graph is interesting as an experiment, but neuron count alone is not a reason to expect better control. A fly-body simulator can help study biological embodiment; the physical drone needs a quadrotor model instead.

## What the five screenshots establish

| Screenshot | Directly visible evidence | What remains unverified |
|---|---|---|
| Alexey Fateev: music/skateboard | “Paint the brain,” stimulation colors, a fly body, audio controls and a displayed 0.04× indicator. The UI resembles Neural Canvas. | Exact source/version and whether the skateboard extension uses learned control. Matching interface text is not proof of a shared codebase. Public-repository searches of the author's account did not locate this extension. |
| Lyra: Beat Saber | 165,122 neurons, 25.56M connections, 135 foreleg motor neurons; the viewport says “motor rehearsal.” | Training implementation, checkpoint, autonomous visual play, performance on unfamiliar maps. No exact public training repository was located. The screenshot supports a rehearsal interpretation; it does not prove closed-loop mastery. |
| RTVisual: DOOM | The frame explicitly says **138,639 neurons / FlyWire v783 female** and reports model time separately from wall time. | It cannot be assumed to be the current MaleCNS DOOMFLY repository. The post text and the embedded video may refer to different versions or projects. Source identity remains unresolved. |
| Dhruv Bhatia: YMCA | Four tone buttons, “recorded model activations,” and “standing · scripted wing flutter.” | Exact weights changed, training/test split and causal dependence on the circuit. No matching public YMCA training release was found in the author's public repositories. The statement about LoVP92 and “love” is not established by this image. |
| Eris: text through legs | Character images → sampled visual fields → connectome → motor-neuron decoder; output text and an articulated body. | Exact code, corpus, held-out loss, decoder capacity and comparison with a decoder-only baseline. A text decoder can be an interesting task adapter; this screenshot does not establish language understanding. |

I inspected the images supplied by the user, not complete videos. Mirrored social posts were used as discovery leads only; technical conclusions below rely on primary documentation, papers and code. Failed searches mean “not located in this review,” not “does not exist.” No claims about experience or consciousness can be inferred from these displays.

**Correction to the earlier plan:** DOOMFLY remains relevant, but the supplied DOOM screenshot does not match its current dataset description. The previous association should not be treated as a verified attribution.

## Most useful findings

### Google/Janelia provides the map, not a ready controller

Google's September 2026 account describes MaleCNS as a reconstruction spanning brain, optic lobes and ventral nerve cord, with over 166,000 neurons and approximately 125 million synaptic contacts. It links the Cell paper and public data. This extends the anatomical substrate for studying sensory-to-motor pathways; it does not supply a trained drone policy. Preserve dataset versions and distinguish individual contacts from aggregated directed edges. [Google Research](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/), [official data](https://male-cns.janelia.org/download/).

### A full graph in a browser is technically credible

Neural Canvas publishes a whole-MaleCNS spiking engine with JavaScript and WebGPU backends. I inspected its model card, GPU allocation code, propagation shader and motion decoder. GPU state persists between steps; sparse propagation follows emitted spikes. Its decoder contains explicit movement and flight rules. The documented body animation has no sensory feedback into the circuit. This is useful engine/UI infrastructure, not an independently validated controller. The exact relationship to Alexey's extended demo is unverified. [README](https://huggingface.co/spaces/Xenova/fruit-fly-simulation/blob/main/README.md), [GPU implementation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation/blob/main/src/brain-gpu.js).

### FlyVis offers a better trained visual starting point

FlyVis combines measured visual connectivity with task optimization and checks predicted neural responses against experiments. The published model has 45,669 neurons, roughly 1.5 million connections and 734 fitted parameters through shared cell-type structure. It targets optical-flow estimation and reproduces important contrast and motion selectivity. It uses earlier visual reconstructions, not the new complete MaleCNS graph. This makes it a valuable separate baseline rather than a drop-in replacement of our data. [Paper](https://www.nature.com/articles/s41586-024-07939-3), [official implementation](https://github.com/TuragaLab/flyvis).

### FlyGM closely matches the architectural-learning question

FlyGM treats a FlyWire-derived graph as a message-passing policy, using imitation learning followed by reinforcement learning for a simulated fly. The authors report comparisons with rewired, random, MLP and other graph models. This supports investigating connectome topology as an architectural prior; it is not evidence that an unchanged physiological brain spontaneously learns the task. Its project page currently lists code as forthcoming. The paper discusses A100 80 GB budgets for graph comparisons, which is far beyond this Mac's training budget. [Preprint and methods](https://arxiv.org/html/2602.17997v3), [project page](https://lnsgroup.cc/research/FlyGM/).

### Embodiment and brain reconstruction are separate components

NeuroMechFly/FlyGym provides sensory feedback, contact dynamics and controlled embodied experiments. The Google DeepMind/Janelia flybody work provides detailed body physics with learned locomotion controllers. These are strong resources for testing fly-like sensorimotor loops; a detailed fly mesh does not imply that its controller is a reconstructed brain. For our hardware task, use a quadrotor simulator and preserve a common controller API. [NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y), [flybody](https://github.com/TuragaLab/flybody), [body-physics paper](https://www.nature.com/articles/s41586-025-09029-4).

## Survey catalogue

“Published” refers to the paper, not independent reproduction here. “Open project” means source is available; functionality and performance remain the author's claims unless explicitly inspected above. Runtime notes are screening judgments.

| # | Work / primary source | Evidence and relevance | Local/reuse assessment |
|---|---|---|---|
| 1 | [Google MaleCNS research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) | Official account of the reconstruction and companion research. | Provenance/context, not simulation software. |
| 2 | [MaleCNS official data](https://male-cns.janelia.org/download/) | Anatomical graph and annotations; already used in our lab. | Retain IDs, contact counts, transformations and hashes. |
| 3 | [Male visual-system inventory](https://research.google/pubs/connectome-driven-neural-inventory-of-a-complete-visual-system/) | Published inventory of right optic-lobe types. | Guides a more defensible visual subgraph and retinotopic mapping. |
| 4 | [Shiu et al. brain model](https://www.nature.com/articles/s41586-024-07763-9) / [code](https://github.com/philshiu/Drosophila_brain_model) | Published LIF sensorimotor model with feeding/grooming circuit tests. | Reference dynamics and perturbation experiments; not a navigation policy. |
| 5 | [FlyVis](https://github.com/TuragaLab/flyvis) / [paper](https://www.nature.com/articles/s41586-024-07939-3) | Published task-optimized mechanistic vision with pretrained models. | Priority for image/optic-flow experiments; isolate its older Python dependencies. |
| 6 | [NeuroMechFly v2 / FlyGym](https://github.com/NeLy-EPFL/flygym) / [paper](https://www.nature.com/articles/s41592-024-02497-y) | Published embodied sensory/control framework, including navigation experiments. | Useful biological comparison environment; test Mac rendering separately. |
| 7 | [NeuroMechFly workshop](https://github.com/NeLy-EPFL/neuromechfly-workshop) | Official runnable tutorials for kinematic replay and fly-following. | Low-cost entry point before integrating a new neural model. |
| 8 | [flybody](https://github.com/TuragaLab/flybody) / [paper](https://www.nature.com/articles/s41586-025-09029-4) | Published physics body, walking/flight tasks and RL controllers. | Use for fly embodiment; training stack is heavier than visualization. |
| 9 | [FlyGM](https://arxiv.org/abs/2602.17997) / [project](https://lnsgroup.cc/research/FlyGM/) | Preprint reporting learned whole-graph locomotion and topology comparisons. | Strong methods reference; exact code release not located, large training budget. |
| 10 | [Connectome to effectome](https://www.nature.com/articles/s41586-024-07982-0) | Published work connecting anatomy to effective neural interactions. | Reminds us that structural weights and causal influence differ. |
| 11 | [Central-complex connectome, Hulse et al.](https://pubmed.ncbi.nlm.nih.gov/34696823/) | Published navigation/action-selection circuit organization. | Priority when selecting heading and goal-direction pathways. |
| 12 | [Allocentric travelling direction via vector computation](https://www.nature.com/articles/s41586-021-04067-0) | Published central-complex vector computation. | Relevant to heading versus travel direction during drone drift. |
| 13 | [Mushroom-body reinforcement prediction errors](https://www.nature.com/articles/s41467-021-22592-4) / [code](https://github.com/BrainsOnBoard/paper_RPEs_in_drosophila_mb) | Published reinforcement-learning model with conditioning analyses. | Start plasticity tests here; MATLAB reference code, not a drone controller. |
| 14 | [Neural Canvas](https://huggingface.co/spaces/Xenova/fruit-fly-simulation/blob/main/README.md) | Open full-MaleCNS browser simulator; selected implementation inspected. | Priority WebGPU engine/visualization benchmark; movement layer must be replaced for control. |
| 15 | [DOOMFLY](https://github.com/nftechie/doomfly) | Open full-MaleCNS game experiment; current v6 reports failed learning gates. | Useful provenance, negative results and validation design; no demonstrated learned survival. |
| 16 | [ornata/fly — Mario](https://github.com/ornata/fly) | Open sensory/network/game loop; explicitly no learning or reward. | Mac-tested by author; useful timestamps, replay and stale-command behavior. |
| 17 | [Help the Fly Escape](https://github.com/dzhng/fly-escape) | Open cropped-connectome game, Rust/WASM, local playback. | Good browser packaging; movement effects do not establish trained navigation. |
| 18 | [Chreatures](https://github.com/emberian/chreatures/blob/main/README.md) | Open artificial-life project with whole-MaleCNS recurrence and modeled learning. | Interesting Mac precedent; distinguish current engine from frozen recordings and sensory bypasses. |
| 19 | [NeuroCraft Fly](https://github.com/evnsnclr/neurocraft-fly-public) | Public project home and release roadmap. | Do not budget against a complete runnable release until source is available. |
| 20 | [Eon fly-brain](https://github.com/eonsystemspbc/fly-brain) | Open FlyWire LIF backend comparisons and spike exports. | Useful CPU/GPU numerical benchmarking; CUDA backends do not transfer directly to Apple GPU. |
| 21 | [lixiang1076/fly-brain](https://github.com/lixiang1076/fly-brain) | Open derivative advertising interaction and modeled dopamine learning. | Screened project lead; no independently established general-control evidence in this review. |
| 22 | [FlyBrainLab](https://github.com/FlyBrainLab/FlyBrainLab) | Interactive anatomical/executable circuit platform. | Reuse analysis ideas; full backend expects Ubuntu/CUDA and recommends much more RAM. |
| 23 | [Neurokernel](https://github.com/neurokernel) | Modular fly neural simulation, retina and vision tools. | Mature infrastructure reference; CUDA-oriented execution needs a different runtime on this Mac. |
| 24 | [bigclust2](https://github.com/flyconnectome/bigclust2) | Connectomic clustering and linked 3D exploration. | Useful for circuit selection and comparative anatomy, not control. |
| 25 | [coconatfly](https://github.com/natverse/coconatfly/blob/master/README.Rmd) | Cross-connectome data access across MaleCNS, FlyWire and others. | Useful annotation reconciliation; R workflow. |
| 26 | [connectome_data_prep](https://github.com/YijieYin/connectome_data_prep) | Sparse graph preparation and axon/dendrite splits. | Data-processing reference; check version and transformation before reuse. |
| 27 | [NanoFlowNet](https://github.com/tudelft/nanoflownet) | Published compact optical flow with real nano-quadcopter obstacle-avoidance demonstration. | Strong non-connectome visual baseline; different camera/compute hardware. |
| 28 | [Learning visual appearance for optical-flow control](https://mavlab.tudelft.nl/learning-process-solution-fundamental-optical-flow-problems/) | Research-lab account of published learning for visual flight control. | Directly relevant to limitations of flow-only distance/landing control. |
| 29 | [Finding the gap](https://www.nature.com/articles/s41467-024-45063-y) | Published insect-inspired spiking motion vision and closed-loop avoidance. | Useful intermediate model; real validation includes a wheeled robot, not our quadrotor. |
| 30 | [Visual route following for tiny autonomous robots](https://pure.tudelft.nl/ws/portalfiles/portal/217003212/scirobotics.adk0310.pdf) | Published insect-inspired route following on a 56 g drone. | Good navigation baseline; panoramic sensing and odometry differ from this UFO. |
| 31 | [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | Open quadrotor control/RL environment; README reports Apple Silicon testing. | Preferred next drone simulator, with parameters fitted to our craft. |
| 32 | [safe-control-gym](https://github.com/learnsyslab/safe-control-gym) | Published control benchmark with disturbances, constraints and conventional/RL controllers. | Useful control baselines and stress tests; more setup than the minimal simulator. |

Additional discovery leads: [insect-inspired route following](https://doi.org/10.1007/s42235-025-00695-8), [Eon embodied-emulation announcement](https://eon.systems/updates/embodied-brain-emulation), and [FlyWire reconstruction](https://doi.org/10.1038/s41586-024-07558-y). Some primary pages were inaccessible or poorly extracted, so they are not counted as closely reviewed implementations. The MaleCNS main Cell paper DOI was resolved through Google's article, but the publisher page could not be retrieved: [10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015).

## What this changes for our Mac setup

### Whole-graph inference benchmark: move earlier

The inspected WebGPU code allocates one graph buffer of `(2N + 1 + E) × 4` bytes and a separate contact-count buffer of `E × 4` bytes. At N=166,700 and E=25,582,938 these are approximately **99 MiB and 98 MiB**, before neural state, history, CPU staging, assets and browser overhead. These calculations make inference plausible; they are not an estimate of total peak memory. The implementation checks storage-buffer limits. [Allocation code](https://huggingface.co/spaces/Xenova/fruit-fly-simulation/blob/main/src/brain-gpu.js).

Benchmark one isolated instance on the M3, recording peak memory pressure, compile/load time, sustained simulated-seconds per wall-second, p50/p95 step latency, activity-dependent workload and CPU/GPU agreement. Keep render FPS separate from neural simulation speed. Do not start gradient training before establishing these costs.

Our 8 GB system may hold compact inference buffers yet still be unable to train large recurrent graphs efficiently. Backpropagation requires activations, gradients and optimizer state; batch size and temporal unrolling matter. A general GPU recommendation cannot be inferred from successful playback.

### Separate Python environments

Our existing app uses Python 3.14. FlyVis declares Python >=3.9 and <3.13; its optional pretrained dependency pins include older packages. Create an isolated environment matching the chosen upstream tutorial/checkpoint, rather than forcing it into the working lab environment. CPU inference is the first reproducibility target; test Apple MPS operator support and numerical agreement before claiming GPU support. [Package configuration](https://github.com/TuragaLab/flyvis/blob/main/pyproject.toml).

### Three independent clocks

The UX should display camera capture/arrival time, neural model time and actuator dispatch time. Add stale-frame age, dropped frames, queue depth, simulated-real-time ratio and checkpoint hash. A smoothly rendered body or recorded sequence can hide a neural simulation that runs well below real time. The supplied screenshots already expose different model/wall times and speed indicators.

## Proposed implementation order

1. **Add benchmark adapters** for the current small rate model and whole-MaleCNS WebGPU model. Preserve provenance and publish CPU/GPU numerical comparisons. No body controller is needed for this benchmark.
2. **Reproduce trained visual responses** with FlyVis on simple gratings, edges and moving dots before using drone footage. Evaluate against a compact conventional flow baseline.
3. **Replace the current arena's proximity brightness** with rendered camera frames in a calibrated quadrotor environment. Otherwise our current training/replay domain mismatch dominates any circuit comparison.
4. **Build an interchangeable policy interface:** timestamped observation in; desired velocity/yaw or bounded high-level action out; learned navigation separated from stabilization. This avoids pretending that fly wing/leg readouts are motor commands for a quadrotor.
5. **Train with matched budgets**: small direct policy, measured graph, rewired graph, shuffled input mapping and no-vision/open-loop controls. Use multiple independent seeds and unseen layouts. Compare both fixed-readout interventions and separately retrained variants.
6. **Study internal plasticity separately** using a conditioning task and an explicit mushroom-body learning rule, with paired/unpaired reward, reversal and plasticity-disabled controls. Then ask whether it improves exploration.
7. **Validate hardware transfer** through recorded footage, live shadow proposals and calibrated bounded flight trials. The current motor protocol is useful plumbing; camera latency and uncertain position are still major limitations.

## UX changes worth borrowing

- Whole-CNS anatomical view with brain/nerve-cord toggles, published positions, region selection and body-ID inspection.
- Separate displays for actual sensory input, stimulated cells, propagated activity and decoded action. Label authored stimulation and direct motor input clearly.
- A causal-inspection mode: silence a region, disable recurrence, shuffle channels or freeze learning, then replay identical inputs.
- Synchronized behavior and neural timelines with model-time/wall-time labels and live-versus-recorded status.
- An experiment ledger with exact model version, trainable parameter count, dataset selection, seed, teacher/reward inputs and held-out metrics.
- Show the drone camera and simulated quadrotor for drone experiments. An optional fly body belongs in a separate embodiment experiment, with its controller type explicit.

## Evidence needed for a publishable claim

“Uses real wiring,” “changes weights,” “learns a training task,” “generalizes,” and “matches biological responses” are separate claims. Give each its own test. For drone work, the defensible initial question is whether a specified connectome-derived architecture improves sample efficiency, robustness or interpretability under controlled sensing and training conditions. The paper should include negative results, workload costs and all engineered adapters.

## Review artifacts and limits

Selected Neural Canvas files are saved under `source-excerpts/neural-canvas/` with source URLs and SHA256 hashes in `source-lock.json`; these are research snapshots, not imported application code. Repository licenses must be checked for the exact files before incorporation; mixed data/body/code licensing occurs in these projects.

Public searches covered the supplied authors/demos, MaleCNS/FlyWire simulation, connectome-constrained vision, embodied fly control, central-complex navigation, mushroom-body learning, browser neural engines and insect-inspired drone navigation. Exact Beat Saber, YMCA, skateboard-extension and Eris training code was not located. No conclusion here depends on accepting the social captions as experimental validation.
