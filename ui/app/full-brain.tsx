'use client';
/* Canvas is the interactive scientific visualization; its role documents the control. */
/* oxlint-disable jsx-a11y/prefer-tag-over-role, jsx-a11y/no-interactive-element-to-noninteractive-role */
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Play, Pause, RotateCcw, SkipForward, Zap } from 'lucide-react';
const API = 'http://127.0.0.1:8766/api/full';
type Frame = {
  ready: boolean;
  status: string;
  running: boolean;
  error: string | null;
  frame: number;
  spikes: number[][];
  model_ms: number;
  batch_ms?: number;
  realtime_ratio?: number;
  active?: number;
  total_spikes?: number;
  stimulus?: string;
  stimulus_hz?: number;
  stimulated?: number;
  recurrence?: boolean;
  group_spikes?: Record<string, number>;
  presets?: Record<string, number>;
  trace: { time: number; total: number; active: number }[];
  adapter?: { device: string; backend_type: string };
  manifest?: {
    nodes: number;
    edges: number;
    contacts: number;
    positions: number;
    classes: string[];
    dynamics: string;
  };
};
type Neuron = {
  index: number;
  id: string;
  type: string;
  group: string;
  side: string | null;
  transmitter: string;
  sign: number;
  soma: number[] | null;
};
const vertex = `#version 300 es
in vec4 point;in float count;uniform vec2 rotation;uniform float zoom;uniform float aspect;uniform float pixelRatio;uniform int selected;
out float glow;out float picked;
void main(){
 vec3 p=vec3(point.x,-point.z,point.y);
 float c=cos(rotation.x),s=sin(rotation.x);p=vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);
 c=cos(rotation.y);s=sin(rotation.y);p=vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);
 gl_Position=vec4(p.x*zoom/aspect,p.y*zoom,p.z*.2,1.0);
 if(point.x>1000.0)gl_Position=vec4(3.0,3.0,0.0,1.0);
 glow=min(1.0,count/3.0);picked=gl_VertexID==selected?1.0:0.0;
 gl_PointSize=(picked>0.0?9.0:count>0.0?3.0+glow*2.0:1.05)*pixelRatio;
}`;
const fragment = `#version 300 es
precision highp float;in float glow;in float picked;out vec4 color;
void main(){float d=length(gl_PointCoord-vec2(.5));if(d>.5)discard;
vec3 ink=mix(vec3(.30,.47,.58),vec3(1.0,.66,.20),step(.01,glow));
if(picked>0.0)ink=vec3(.75,1.0,.5);
color=vec4(ink,(glow>0.0?.85:.21)*(1.0-d));}`;
export function BrainCanvas({
  positions,
  spikes,
  selected,
  onSelect,
}: {
  positions: Float32Array | null;
  spikes: number[][];
  selected: number;
  onSelect: (i: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null),
    camera = useRef({ yaw: 0, pitch: 0, zoom: 0.91 }),
    draw = useRef(() => {}),
    upload = useRef<(a: Float32Array) => void>(() => {}),
    selection = useRef(selected),
    drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(
      null,
    );
  const [error, setError] = useState('');
  useEffect(() => {
    const c = canvas.current;
    if (!c || !positions) return;
    const gl = c.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) {
      queueMicrotask(() => setError('WebGL2 is unavailable in this browser.'));
      return;
    }
    const activateProgram = gl.useProgram.bind(gl);
    const shaders: WebGLShader[] = [];
    const compile = (type: number, code: string) => {
      const shader = gl.createShader(type)!;
      shaders.push(shader);
      gl.shaderSource(shader, code);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw Error(gl.getShaderInfoLog(shader) || 'Shader compilation failed');
      return shader;
    };
    const program = gl.createProgram()!;
    let data: WebGLBuffer | null = null,
      activity: WebGLBuffer | null = null,
      observer: ResizeObserver | null = null;
    try {
      gl.attachShader(program, compile(gl.VERTEX_SHADER, vertex));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw Error(gl.getProgramInfoLog(program) || 'Shader link failed');
      activateProgram(program);
      data = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, data);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(program, 'point');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
      activity = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, activity);
      gl.bufferData(gl.ARRAY_BUFFER, positions.length, gl.DYNAMIC_DRAW);
      const a = gl.getAttribLocation(program, 'count');
      gl.enableVertexAttribArray(a);
      gl.vertexAttribPointer(a, 1, gl.FLOAT, false, 0, 0);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.clearColor(0.025, 0.046, 0.06, 1);
      draw.current = () => {
        const box = c.getBoundingClientRect(),
          dpr = Math.min(devicePixelRatio, 2);
        c.width = Math.max(1, Math.round(box.width * dpr));
        c.height = Math.max(1, Math.round(box.height * dpr));
        gl.viewport(0, 0, c.width, c.height);
        activateProgram(program);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform2f(
          gl.getUniformLocation(program, 'rotation'),
          camera.current.yaw,
          camera.current.pitch,
        );
        gl.uniform1f(
          gl.getUniformLocation(program, 'zoom'),
          camera.current.zoom,
        );
        gl.uniform1f(
          gl.getUniformLocation(program, 'aspect'),
          c.width / c.height,
        );
        gl.uniform1f(gl.getUniformLocation(program, 'pixelRatio'), dpr);
        gl.uniform1i(
          gl.getUniformLocation(program, 'selected'),
          selection.current,
        );
        gl.drawArrays(gl.POINTS, 0, positions.length / 4);
      };
      upload.current = (counts) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, activity);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, counts);
        draw.current();
      };
      observer = new ResizeObserver(() => draw.current());
      observer.observe(c);
      draw.current();
    } catch (e) {
      queueMicrotask(() => setError(String(e)));
    }
    return () => {
      observer?.disconnect();
      draw.current = () => {};
      upload.current = () => {};
      gl.deleteBuffer(data);
      gl.deleteBuffer(activity);
      gl.deleteProgram(program);
      shaders.forEach((s) => gl.deleteShader(s));
    };
  }, [positions]);
  useEffect(() => {
    if (!positions) return;
    const counts = new Float32Array(positions.length / 4);
    for (const [i, n] of spikes) counts[i] = n;
    upload.current(counts);
  }, [positions, spikes]);
  useEffect(() => {
    selection.current = selected;
    draw.current();
  }, [selected]);
  const pick = (x: number, y: number) => {
    if (!positions || !canvas.current) return;
    const b = canvas.current.getBoundingClientRect(),
      cam = camera.current,
      aspect = b.width / b.height;
    let nearest = 100,
      index = -1;
    for (let i = 0; i < positions.length; i += 4) {
      if (positions[i] > 1000) continue;
      const px = positions[i],
        py = -positions[i + 2],
        pz = positions[i + 1];
      const xx = Math.cos(cam.yaw) * px + Math.sin(cam.yaw) * pz,
        zz = -Math.sin(cam.yaw) * px + Math.cos(cam.yaw) * pz;
      const yy = Math.cos(cam.pitch) * py - Math.sin(cam.pitch) * zz;
      const sx = b.left + b.width * (0.5 + (xx * cam.zoom) / aspect / 2),
        sy = b.top + b.height * (0.5 - (yy * cam.zoom) / 2);
      const distance = (sx - x) ** 2 + (sy - y) ** 2;
      if (distance < nearest) {
        nearest = distance;
        index = i / 4;
      }
    }
    if (index >= 0) onSelect(index);
  };
  return (
    <div className="full-canvas-wrap">
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <canvas
        ref={canvas}
        tabIndex={0}
        role="application"
        aria-label="Full nervous system. Drag or use arrow keys to rotate. Plus and minus zoom. Click a neuron to inspect."
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') camera.current.yaw -= 0.1;
          else if (e.key === 'ArrowRight') camera.current.yaw += 0.1;
          else if (e.key === 'ArrowUp') camera.current.pitch += 0.1;
          else if (e.key === 'ArrowDown') camera.current.pitch -= 0.1;
          else if (e.key === '+') camera.current.zoom *= 1.1;
          else if (e.key === '-') camera.current.zoom /= 1.1;
          else return;
          e.preventDefault();
          draw.current();
        }}
        onPointerDown={(e) => {
          drag.current = {
            x: e.clientX,
            y: e.clientY,
            yaw: camera.current.yaw,
            pitch: camera.current.pitch,
          };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const p = drag.current;
          if (p) {
            camera.current.yaw = p.yaw + (e.clientX - p.x) / 220;
            camera.current.pitch = p.pitch + (e.clientY - p.y) / 220;
            draw.current();
          }
        }}
        onPointerUp={(e) => {
          const p = drag.current;
          if (p && Math.hypot(e.clientX - p.x, e.clientY - p.y) < 5)
            pick(e.clientX, e.clientY);
          drag.current = null;
        }}
        onWheel={(e) => {
          camera.current.zoom = Math.max(
            0.25,
            Math.min(5, camera.current.zoom * Math.exp(-e.deltaY * 0.001)),
          );
          draw.current();
        }}
      />
      <div className="full-view-buttons">
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            camera.current = { yaw: 0, pitch: 0, zoom: 0.91 };
            draw.current();
          }}
        >
          Front
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            camera.current = { yaw: Math.PI / 2, pitch: 0, zoom: 0.91 };
            draw.current();
          }}
        >
          Side
        </Button>
        <span>Drag to orbit · scroll to zoom · click a neuron</span>
      </div>
    </div>
  );
}
export default function FullBrain() {
  const [frame, setFrame] = useState<Frame | null>(null),
    [positions, setPositions] = useState<Float32Array | null>(null),
    [neuron, setNeuron] = useState<Neuron | null>(null),
    [hz, setHz] = useState(120),
    [preset, setPreset] = useState('LC4'),
    [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const r = await fetch(`${API}/state`);
        if (!r.ok) throw Error('Full-brain service unavailable');
        const d = (await r.json()) as Frame;
        if (active) {
          setFrame(d);
          setError('');
        }
      } catch (e) {
        if (active) setError(String(e));
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 400);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (!frame?.ready || positions) return;
    let active = true;
    const get = async () => {
      try {
        const r = await fetch(`${API}/positions`);
        if (!r.ok) throw Error('Anatomical positions unavailable');
        const b = await r.arrayBuffer();
        if (active) setPositions(new Float32Array(b));
      } catch (e) {
        if (active) setError(String(e));
      }
    };
    void get();
    const timer = setInterval(() => void get(), 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [frame?.ready, positions]);

  const command = async (action: string, extra: object = {}) => {
    try {
      setError('');
      const r = await fetch(`${API}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...extra }),
      });
      if (!r.ok) throw Error(await r.text());
    } catch (e) {
      setError(String(e));
    }
  };
  const inspect = async (index: number) => {
    try {
      const r = await fetch(`${API}/neuron/${index}`);
      if (!r.ok) throw Error('Neuron lookup failed');
      setNeuron((await r.json()) as Neuron);
    } catch (e) {
      setError(String(e));
    }
  };
  const m = frame?.manifest,
    ready = !!frame?.ready && !frame.error,
    trace = frame?.trace || [],
    max = Math.max(1, ...trace.map((t) => t.total));
  return (
    <div className="full-lab">
      <div className="toolbar">
        <div>
          <h2 className="full-title">Full fly nervous system</h2>
          <p className="help">
            Measured MaleCNS wiring · computed spikes · WebGPU /{' '}
            {frame?.adapter?.backend_type || 'Metal'}
          </p>
        </div>
        <div className="toolbar-group">
          <Button
            disabled={!ready}
            onClick={() => void command(frame?.running ? 'pause' : 'run')}
          >
            {frame?.running ? <Pause size={15} /> : <Play size={15} />}{' '}
            {frame?.running ? 'Pause' : 'Run'}
          </Button>
          <Button
            variant="outline"
            disabled={!ready}
            onClick={() => void command('step')}
          >
            <SkipForward size={15} />
            10 ms
          </Button>
          <Button
            variant="ghost"
            disabled={!ready}
            onClick={() => void command('reset')}
          >
            <RotateCcw size={15} />
            Reset
          </Button>
        </div>
      </div>
      {(error || frame?.error) && (
        <div className="error" role="alert">
          {error || frame?.error}
        </div>
      )}
      <div className="full-stats">
        {[
          ['Neurons', m?.nodes.toLocaleString() || '166,700'],
          ['Directed connections', m?.edges.toLocaleString() || '25,582,938'],
          ['Firing this window', frame?.active?.toLocaleString() || '0'],
          ['Neural time', `${frame?.model_ms.toFixed(0) || 0} ms`],
        ].map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong className="mono">{value}</strong>
          </div>
        ))}
      </div>
      <div className="full-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Brain + ventral nerve cord</h2>
            <span className="note mono">{frame?.status || 'CONNECTING'}</span>
          </div>
          <BrainCanvas
            positions={positions}
            spikes={frame?.spikes || []}
            selected={neuron?.index ?? -1}
            onSelect={(i) => void inspect(i)}
          />
          <div className="full-caption">
            <span>
              <i className="rest-key" />
              Resting / no spike in window <i className="fire-key" />
              Fired in last 10 ms of neural time
            </span>
            <span>
              {m?.positions.toLocaleString() || '—'} published soma positions
            </span>
          </div>
          <p className="help full-disclosure">
            All {m?.nodes.toLocaleString() || '166,700'} entries run. Cells
            without published soma positions are omitted from the view. Gold
            points reflect actual spike counts; brightness increases with count.
          </p>
        </section>
        <aside className="side">
          <section className="panel">
            <div className="panel-header">
              <h2>Stimulate the circuit</h2>
              <Zap size={16} />
            </div>
            <div className="panel-body controls">
              <label htmlFor="full-stimulus">Input population</label>
              <Select
                value={preset}
                onValueChange={(v) => v && setPreset(v)}
                items={[
                  { value: 'LC4', label: 'LC4 · visual projection' },
                  { value: 'LC9', label: 'LC9 · visual projection' },
                  { value: 'DNa02', label: 'DNa02 · descending' },
                  { value: 'retina', label: 'R1–R6 · photoreceptors' },
                  { value: 'off', label: 'No external stimulus' },
                ]}
              >
                <SelectTrigger id="full-stimulus">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[
                    ['LC4', 'LC4 · visual projection'],
                    ['LC9', 'LC9 · visual projection'],
                    ['DNa02', 'DNa02 · descending'],
                    ['retina', 'R1–R6 · photoreceptors'],
                    ['off', 'No external stimulus'],
                  ].map(([v, l]) => (
                    <SelectItem value={v} key={v}>
                      {l}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <label htmlFor="full-hz">Poisson input rate · Hz</label>
              <Input
                id="full-hz"
                type="number"
                value={hz}
                min={0}
                max={400}
                onChange={(e) => setHz(Number(e.target.value))}
              />
              <Button
                disabled={!ready}
                onClick={() => void command('stimulus', { preset, hz })}
              >
                Apply stimulus
              </Button>
              <p className="help">
                Active: {frame?.stimulus || 'LC4'} · {frame?.stimulus_hz || 0}{' '}
                Hz · {frame?.stimulated || 0} cells. This is engineered external
                stimulation.
              </p>
              <Button
                variant="outline"
                disabled={!ready}
                onClick={() =>
                  void command('recurrence', { enabled: !frame?.recurrence })
                }
              >
                {frame?.recurrence ? 'Disable' : 'Enable'} recurrent
                transmission
              </Button>
              <p className="help">
                Recurrence {frame?.recurrence ? 'on' : 'off'}. Disabling
                delivery tests whether activity spreads beyond stimulated cells.
              </p>
            </div>
          </section>
          <section className="panel">
            <div className="panel-header">
              <h2>Neuron inspector</h2>
            </div>
            <div className="panel-body">
              {neuron ? (
                <>
                  <strong>{neuron.type}</strong>
                  <p className="mono">Body ID {neuron.id}</p>
                  <p className="help">
                    {neuron.group.replaceAll('_', ' ')} ·{' '}
                    {neuron.side || 'side unknown'}
                    <br />
                    {neuron.transmitter} · modeled sign {neuron.sign}
                  </p>
                  <p className="help">
                    Spikes this window:{' '}
                    {frame?.spikes.find((s) => s[0] === neuron.index)?.[1] || 0}
                  </p>
                  <Button
                    style={{ marginTop: 12 }}
                    disabled={!ready}
                    onClick={() =>
                      void command('neuron', { index: neuron.index, hz })
                    }
                  >
                    Stimulate this neuron
                  </Button>
                </>
              ) : (
                <p className="help">
                  Click a point in the anatomical view to see its identity and
                  firing count.
                </p>
              )}
            </div>
          </section>
          <section className="panel">
            <div className="panel-header">
              <h2>Simulation timing</h2>
            </div>
            <div className="panel-body help">
              {frame?.adapter?.device || 'Loading GPU'}
              <br />
              {frame?.batch_ms?.toFixed(1) || '—'} ms wall time / 10 ms neural
              batch
              <br />
              {frame?.realtime_ratio?.toFixed(3) || '—'}× neural / compute time
              <p style={{ marginTop: 10 }}>
                Playback is not accelerated to hide computation time. This
                spiking model is separate from the earlier rate-model benchmark.
              </p>
            </div>
          </section>
        </aside>
      </div>
      <section className="panel full-trace">
        <div className="panel-header">
          <h2>Network firing</h2>
          <span className="note">
            Total spikes per 10 ms neural window · peak {max.toLocaleString()}
          </span>
        </div>
        <svg
          viewBox="0 0 1000 120"
          role="img"
          aria-label="Actual network spike counts over successive neural time windows"
        >
          <polyline
            fill="none"
            stroke="#efb862"
            strokeWidth="2"
            points={trace
              .map(
                (t, i) =>
                  `${10 + (i / Math.max(1, trace.length - 1)) * 980},${105 - (t.total / max) * 90}`,
              )
              .join(' ')}
          />
          <text x="10" y="118">
            {trace[0]?.time.toFixed(0) || 0} ms
          </text>
          <text x="910" y="118">
            {frame?.model_ms.toFixed(0) || 0} ms
          </text>
        </svg>
      </section>
      <p className="help full-disclosure">
        {m?.dynamics} No motor commands are produced by this mode. Structure is
        measured; dynamics and stimulus mapping are modeling assumptions.
      </p>
    </div>
  );
}
