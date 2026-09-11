'use client';
/* Canvas/SVG visuals require an image role; live JPEG frames must bypass image optimization. */
/* oxlint-disable jsx-a11y/prefer-tag-over-role, next/no-img-element */
import FullBrain from '../full-brain';
import { FlightControls } from '../flight-controls';
import { useEffect, useRef, useState } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Download,
  Network,
  FlaskConical,
  Activity,
  Radio,
  ArrowUpRight,
  Square,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
const API = 'http://127.0.0.1:8766';
const COLORS: Record<string, string> = {
  retina: '#efc36f',
  lamina: '#85d8ef',
  medulla: '#a7ef95',
  motion: '#ccadfa',
};
type Node = {
  id: string;
  type: string;
  group: string;
  hex: number[];
  inhibitory: boolean;
  transmitter: string;
  soma: number[] | null;
  hex_inferred: boolean;
};
type Graph = { nodes: Node[]; edges: number[][] };
type Metrics = { reward: number; coverage: number; collisions: number };
type ArenaState = Metrics & {
  seed: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  time: number;
  distance: number;
  walls: number[][];
  path: number[][];
};
type Sample = Record<string, number>;
type RunResult = { history: Sample[]; evaluation: Metrics; untrained: Metrics };
type State = {
  status: string;
  running: boolean;
  mode: string;
  condition: string;
  seed: number;
  arena: ArenaState;
  rates: number[];
  actions: number[];
  trace: Sample[];
  compute_ms: number;
  training: { status: string; history: Sample[]; generations?: number };
  events: { time: string; message: string }[];
  manifest: {
    nodes: number;
    edges: number;
    contacts: number;
    selection: string;
    sha256: Record<string, string>;
  };
  recording: string;
  replay_time: number;
  replay_duration: number;
  recordings: { id: string; name: string }[];
  last_result: RunResult | null;
  comparison: (Metrics & { condition: string })[];
  live_fresh: boolean;
  live_frame: number;
};
function Choice({
  value,
  onChange,
  items,
  className = '',
  id,
}: {
  id?: string;
  value: string;
  onChange: (s: string) => void;
  items: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <Select value={value} onValueChange={(v) => v && onChange(v)} items={items}>
      <SelectTrigger id={id} className={'section-select ' + className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((i) => (
          <SelectItem key={i.value} value={i.value}>
            {i.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
function Panel({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span className="note mono">{meta}</span>
      </div>
      {children}
    </section>
  );
}
function NeuralView({
  graph,
  rates,
  onSelect,
  selected,
  layout,
}: {
  graph: Graph | null;
  rates: number[];
  onSelect: (i: number) => void;
  selected: number | null;
  layout: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null),
    rot = useRef(0.3),
    drag = useRef<number | null>(null),
    points = useRef<{ x: number; y: number; i: number }[]>([]);
  const [rotation, setRotation] = useState(0.3);
  useEffect(() => {
    const c = ref.current;
    if (!c || !graph) return;
    const draw = () => {
      const box = c.getBoundingClientRect(),
        w = box.width,
        h = box.height,
        dpr = devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = h * dpr;
      const ctx = c.getContext('2d')!;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      const groups = Object.keys(COLORS);
      const raw = graph.nodes
        .map((n, i) => {
          if (layout === 'soma')
            return n.soma ? { i, p: n.soma.map((x) => x / 10000) } : null;
          const layer = groups.indexOf(n.group);
          const q = n.hex[0] - 19,
            r = n.hex[1] - 20;
          const within = (Number(n.id) % 13) / 40;
          return {
            i,
            p: [(layer - 1.5) * 3.3, (q - r * 0.5) * 0.52, r * 0.5 + within],
          };
        })
        .filter(Boolean) as { i: number; p: number[] }[];
      let center = [0, 0, 0],
        scale = 1;
      if (layout === 'soma' && raw.length) {
        center = [0, 1, 2].map(
          (k) => raw.reduce((s, n) => s + n.p[k], 0) / raw.length,
        );
        scale = 2.8;
      }
      const projected = raw.map(({ i, p }) => {
        const x = (p[0] - center[0]) * scale,
          y = (p[1] - center[1]) * scale,
          z = (p[2] - center[2]) * scale;
        const xx = x * Math.cos(rotation) - z * Math.sin(rotation),
          zz = x * Math.sin(rotation) + z * Math.cos(rotation);
        return {
          i,
          x: w / 2 + xx * Math.min(w / 15, h / 11),
          y:
            h / 2 +
            y * Math.min(w / 15, h / 11) * 0.8 +
            zz * Math.min(w / 15, h / 11) * 0.32,
          z: zz,
        };
      });
      points.current = projected;
      const map = new Map(projected.map((p) => [p.i, p]));
      const edges =
        selected === null
          ? graph.edges.filter((_, i) => i % 24 === 0)
          : graph.edges.filter((e) => e[0] === selected || e[1] === selected);
      for (const [a, b] of edges) {
        const p = map.get(a),
          q = map.get(b);
        if (!p || !q) continue;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.strokeStyle =
          selected === null ? 'rgba(156,196,181,.085)' : 'rgba(198,248,128,.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
      for (const p of projected.sort((a, b) => a.z - b.z)) {
        const n = graph.nodes[p.i],
          v = rates[p.i] || 0;
        ctx.globalAlpha = 0.2 + v * 0.8;
        ctx.fillStyle = COLORS[n.group];
        ctx.beginPath();
        ctx.arc(
          p.x,
          p.y,
          selected === p.i ? 5 : 1.15 + v * 1.7,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        if (selected === p.i) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = '#e4ebe8';
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [graph, rates, rotation, selected, layout]);
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Measured connectome nodes, colored by circuit group and brightness by modeled activity. Drag to rotate; select a neuron to inspect its measured connections."
      onPointerDown={(e) => {
        drag.current = e.clientX;
        rot.current = rotation;
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (drag.current !== null)
          setRotation(rot.current + (e.clientX - drag.current) / 160);
      }}
      onPointerUp={(e) => {
        if (drag.current !== null && Math.abs(e.clientX - drag.current) < 5) {
          const box = e.currentTarget.getBoundingClientRect();
          const p = points.current.reduce(
            (best, p) =>
              Math.hypot(
                p.x - e.clientX + box.left,
                p.y - e.clientY + box.top,
              ) < best.d
                ? {
                    i: p.i,
                    d: Math.hypot(
                      p.x - e.clientX + box.left,
                      p.y - e.clientY + box.top,
                    ),
                  }
                : best,
            { i: -1, d: 18 },
          );
          if (p.i >= 0) onSelect(p.i);
        }
        drag.current = null;
      }}
    />
  );
}
function ArenaView({ arena }: { arena: ArenaState }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !arena) return;
    const draw = () => {
      const b = c.getBoundingClientRect(),
        w = b.width,
        h = b.height,
        dpr = devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = h * dpr;
      const ctx = c.getContext('2d')!;
      ctx.scale(dpr, dpr);
      const s = Math.min(w - 70, h - 55) / 12,
        ox = (w - s * 12) / 2,
        oy = (h - s * 12) / 2;
      ctx.clearRect(0, 0, w, h);
      const xy = (x: number, y: number) => [ox + x * s, oy + y * s];
      ctx.strokeStyle = '#1c2b2d';
      ctx.lineWidth = 0.6;
      for (let i = 0; i <= 24; i++) {
        ctx.beginPath();
        ctx.moveTo(ox + (i * s) / 2, oy);
        ctx.lineTo(ox + (i * s) / 2, oy + 12 * s);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ox, oy + (i * s) / 2);
        ctx.lineTo(ox + 12 * s, oy + (i * s) / 2);
        ctx.stroke();
      }
      const unique = new Set<string>();
      for (const [x, y] of arena.path) {
        const key = `${Math.floor(x * 2)},${Math.floor(y * 2)}`;
        if (unique.has(key)) continue;
        unique.add(key);
        ctx.fillStyle = '#203c30';
        ctx.fillRect(
          ox + (Math.floor(x * 2) * s) / 2,
          oy + (Math.floor(y * 2) * s) / 2,
          s / 2,
          s / 2,
        );
      }
      ctx.strokeStyle = '#536466';
      ctx.lineWidth = 3;
      for (const [a, b, c, d] of arena.walls) {
        ctx.beginPath();
        ctx.moveTo(...(xy(a, b) as [number, number]));
        ctx.lineTo(...(xy(c, d) as [number, number]));
        ctx.stroke();
      }
      ctx.strokeStyle = '#c6f880';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      arena.path.forEach(([x, y]: number[], i: number) =>
        i
          ? ctx.lineTo(...(xy(x, y) as [number, number]))
          : ctx.moveTo(...(xy(x, y) as [number, number])),
      );
      ctx.stroke();
      const [x, y] = xy(arena.x, arena.y);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.arc(x, y, s * 2, arena.heading - 1.3, arena.heading + 1.3);
      ctx.closePath();
      ctx.fillStyle = '#c6f8800d';
      ctx.fill();
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(arena.heading);
      ctx.beginPath();
      ctx.moveTo(9, 0);
      ctx.lineTo(-6, -5);
      ctx.lineTo(-3, 0);
      ctx.lineTo(-6, 5);
      ctx.closePath();
      ctx.fillStyle = '#c6f880';
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#7d9490';
      ctx.font = '11px monospace';
      ctx.fillText('0', ox - 16, oy + 5);
      ctx.fillText('12 m', ox + s * 12 - 30, oy + s * 12 + 18);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [arena]);
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Simulated planar arena with obstacles, agent location, explored cells and trajectory"
    />
  );
}
function ActivityChart({
  trace,
  training = false,
}: {
  trace: Sample[];
  training?: boolean;
}) {
  const keys = training ? ['reward', 'mean_reward'] : Object.keys(COLORS);
  const max = training
      ? Math.max(1, ...trace.flatMap((x) => keys.map((k) => x[k])))
      : 1,
    min = training
      ? Math.min(0, ...trace.flatMap((x) => keys.map((k) => x[k])))
      : 0;
  return (
    <svg
      className="chart"
      viewBox="0 0 550 160"
      role="img"
      aria-label={
        training
          ? 'Measured training reward per generation'
          : 'Modeled mean activity over recent steps'
      }
    >
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1="40" y1={130 - v * 105} x2="535" y2={130 - v * 105} />
          <text x="5" y={134 - v * 105}>
            {(min + v * (max - min)).toFixed(training ? 0 : 1)}
          </text>
        </g>
      ))}
      {keys.map((k, i) => (
        <polyline
          key={k}
          fill="none"
          stroke={training ? (i ? '#677d74' : '#c6f880') : COLORS[k]}
          strokeWidth="1.7"
          points={trace
            .map(
              (r, j) =>
                `${40 + (j / Math.max(1, trace.length - 1)) * 495},${130 - ((r[k] - min) / (max - min || 1)) * 105}`,
            )
            .join(' ')}
        />
      ))}
      <text x="40" y="153">
        {training ? 'Generation 1' : 'Earlier'}
      </text>
      <text x="485" y="153">
        {training ? trace.length : 'Now'}
      </text>
    </svg>
  );
}
function Replay({ state }: { state: State }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (
      ref.current &&
      Math.abs(ref.current.currentTime - state.replay_time) > 0.25
    )
      ref.current.currentTime = state.replay_time;
  }, [state.replay_time]);
  return (
    <video
      ref={ref}
      src={`${API}/api/media/${state.recording}`}
      muted
      playsInline
      preload="auto"
      aria-label="Recorded drone footage synchronized to neural input"
    />
  );
}
export default function Home() {
  const [state, setState] = useState<State | null>(null),
    [graph, setGraph] = useState<Graph | null>(null),
    [error, setError] = useState(''),
    [online, setOnline] = useState(false),
    [seed, setSeed] = useState(11),
    [generations, setGenerations] = useState(8),
    [selected, setSelected] = useState<number | null>(null),
    [layout, setLayout] = useState('layers');
  useEffect(() => {
    let active = true;
    const poll = () =>
      fetch(`${API}/api/state`)
        .then((r) => {
          if (!r.ok) throw Error('Worker unavailable');
          return r.json();
        })
        .then((d) => {
          if (active) {
            setState(d as State);
            setOnline(true);
          }
        })
        .catch(() => {
          if (active) setOnline(false);
        });
    void poll();
    const t = setInterval(poll, 250);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, []);
  useEffect(() => {
    if (online && !graph)
      fetch(`${API}/api/graph`)
        .then((r) => r.json())
        .then((d) => setGraph(d as Graph))
        .catch(() => {});
  }, [online, graph]);
  const command = async (action: string, extra: object = {}) => {
    setError('');
    try {
      const r = await fetch(`${API}/api/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, seed, ...extra }),
      });
      const d = (await r.json()) as State;
      if (!r.ok)
        throw Error(
          (d as unknown as { detail?: string }).detail || 'Command failed',
        );
      setState(d);
    } catch (e) {
      setError(String(e));
    }
  };
  const busy =
      !!state && ['training', 'comparing'].includes(state.training.status),
    manifest = state?.manifest,
    n = selected !== null ? graph?.nodes[selected] : null;
  return (
    <main className="shell">
      <header className="mast">
        <div className="brand">
          <div className="brand-mark">
            <Network size={23} />
          </div>
          <div>
            <h1>
              fly<span style={{ color: '#3dd68c' }}>By</span>Wire
            </h1>
            <p>
              <a href="/">Workspace</a> / Research tools
            </p>
          </div>
        </div>
        <div className={'connection mono ' + (!online ? 'offline' : '')}>
          <span className="dot" />
          {online ? 'LOCAL WORKER CONNECTED' : 'WORKER OFFLINE'}
          <span style={{ color: '#4c5e58' }}> / </span>M3 · 8 GB
        </div>
      </header>
      <FlightControls />
      <Tabs defaultValue="full" className="tabs-main">
        <div className="navrow">
          <TabsList variant="line">
            <TabsTrigger value="full">
              <Network size={15} />
              Full brain
            </TabsTrigger>
            <TabsTrigger value="experiment">
              <Activity size={15} />
              Experiment
            </TabsTrigger>
            <TabsTrigger value="results">
              <FlaskConical size={15} />
              Results
            </TabsTrigger>
            <TabsTrigger value="methods">
              <Network size={15} />
              Methods & setup
            </TabsTrigger>
          </TabsList>
          <span className="badge mono">
            <span className="dot" /> BRAIN OUTPUTS: OBSERVATION ONLY
          </span>
        </div>
        {error && (
          <div role="alert" className="error">
            {error}
          </div>
        )}
        {!online && (
          <div role="status" className="error">
            The local worker is offline. Start the lab with{' '}
            <code>./start-lab.sh</code>. Brain outputs remain observational.
          </div>
        )}
        <TabsContent value="full">
          <FullBrain />
        </TabsContent>
        <TabsContent value="experiment">
          <div className="toolbar">
            <div className="toolbar-group">
              <Choice
                value={state?.mode || 'simulation'}
                onChange={(mode) => command('mode', { mode })}
                items={[
                  { value: 'simulation', label: 'Simulation arena' },
                  { value: 'replay', label: 'Drone recording' },
                  { value: 'live', label: 'Live camera · observe only' },
                ]}
              />
              <Button
                disabled={!online}
                onClick={() => command(state?.running ? 'pause' : 'run')}
              >
                {state?.running ? <Pause size={15} /> : <Play size={15} />}{' '}
                {state?.running ? 'Pause' : 'Run'}
              </Button>
              <Button
                variant="outline"
                disabled={!online}
                onClick={() => command('step')}
                aria-label="Advance one model step"
              >
                <SkipForward size={15} />
              </Button>
              <Button
                variant="ghost"
                disabled={!online}
                onClick={() => command('reset')}
              >
                <RotateCcw size={15} />
                Reset
              </Button>
            </div>
            <span className="note mono">
              {state?.status || 'CONNECTING'} ·{' '}
              {state?.compute_ms?.toFixed(2) || '—'} ms / step
            </span>
          </div>
          {state?.mode === 'replay' && (
            <div style={{ marginBottom: 18 }}>
              <Choice
                className="clipselect"
                value={state.recording}
                onChange={(recording) =>
                  command('mode', { mode: 'replay', recording })
                }
                items={state.recordings.map((r) => ({
                  value: r.id,
                  label: r.name,
                }))}
              />
            </div>
          )}
          <div className="workbench">
            <div className="maincol">
              <div className="visuals">
                <Panel
                  title={
                    state?.mode === 'simulation'
                      ? 'Exploration arena'
                      : state?.mode === 'live'
                        ? 'Drone camera · observe only'
                        : 'Recorded flight'
                  }
                  meta={
                    state?.mode === 'simulation'
                      ? '12 × 12 m · planar model'
                      : state?.mode === 'live'
                        ? '160 × 120 · rotated CW'
                        : '320 × 240 · rotated CW'
                  }
                >
                  <div className="viewport">
                    {state?.mode === 'simulation' ? (
                      <ArenaView arena={state.arena} />
                    ) : state?.mode === 'replay' ? (
                      <Replay state={state} />
                    ) : state?.mode === 'live' ? (
                      state.live_fresh ? (
                        <img
                          src={`${API}/api/live.jpg?t=${state.live_frame}`}
                          alt="Live camera, no control outputs"
                        />
                      ) : (
                        <div className="empty">
                          <Radio size={25} />
                          <p>
                            Waiting for the drone camera.
                            <br />
                            Join its Wi-Fi, then select this source again.
                          </p>
                        </div>
                      )
                    ) : (
                      <div className="empty">Waiting for experiment state</div>
                    )}
                    <span className="viewport-label mono">
                      {state?.mode === 'simulation'
                        ? 'SIMULATED EXPLORATION'
                        : 'NEURAL INPUT · COMMANDS NOT SENT'}
                    </span>
                    <div className="viewport-foot mono">
                      <span>
                        {state?.mode === 'simulation'
                          ? `SEED ${state.seed}`
                          : state?.mode === 'replay'
                            ? `${state.replay_time.toFixed(1)} / ${state.replay_duration.toFixed(1)} s`
                            : 'RTSP / UDP'}
                      </span>
                      <span>
                        {state?.mode === 'simulation'
                          ? `${state.arena.time.toFixed(1)} s SIM TIME`
                          : ''}
                      </span>
                    </div>
                  </div>
                  <div className="view-footer">
                    {state?.mode === 'simulation'
                      ? 'Novel cells earn reward. Wall contact is penalized. Flight height and rotor dynamics are not modeled.'
                      : 'Video drives the same modeled retina. Readouts are proposals, not calibrated flight commands.'}
                  </div>
                </Panel>
                <Panel
                  title="Circuit activity"
                  meta={`${manifest?.nodes?.toLocaleString() || '—'} neurons`}
                >
                  <div className="viewport">
                    <NeuralView
                      graph={graph}
                      rates={state?.rates || []}
                      selected={selected}
                      onSelect={setSelected}
                      layout={layout}
                    />
                    <span className="viewport-label mono">
                      {layout === 'layers'
                        ? 'SCHEMATIC CIRCUIT LAYERS'
                        : 'PUBLISHED SOMA COORDINATES'}
                    </span>
                    <div className="viewport-foot">
                      <span>Drag to orbit · click to inspect</span>
                      <span>Brightness = rate</span>
                    </div>
                  </div>
                  <div className="view-footer">
                    <div className="legend">
                      {Object.entries(COLORS).map(([g, c]) => (
                        <span key={g}>
                          <i style={{ background: c }} />
                          {g}
                        </span>
                      ))}
                    </div>
                    <span>Edges are measured. Dynamics are modeled.</span>
                  </div>
                  {n && (
                    <div className="selected">
                      <div className="row">
                        <b>
                          {n.type} · {n.id}
                        </b>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelected(null)}
                        >
                          Clear
                        </Button>
                      </div>
                      {n.transmitter} · rate{' '}
                      {(state?.rates[selected!] || 0).toFixed(3)} ·{' '}
                      {n.hex_inferred ? 'inferred' : 'published'} hex position
                    </div>
                  )}
                </Panel>
              </div>
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Explored cells</span>
                  <div className="number mono">
                    {state?.mode === 'simulation' ? state.arena.coverage : '—'}
                    <small>0.5 m grid</small>
                  </div>
                </div>
                <div className="metric">
                  <span className="metric-label">Episode reward</span>
                  <div className="number mono">
                    {state?.mode === 'simulation'
                      ? state.arena.reward.toFixed(1)
                      : '—'}
                  </div>
                </div>
                <div className="metric">
                  <span className="metric-label">Wall-contact steps</span>
                  <div className="number mono">
                    {state?.mode === 'simulation'
                      ? state.arena.collisions
                      : '—'}
                  </div>
                </div>
                <div className="metric">
                  <span className="metric-label">Measured connections</span>
                  <div className="number mono">
                    {manifest?.edges?.toLocaleString() || '—'}
                  </div>
                </div>
              </div>
              <div className="lower">
                <Panel title="Population response" meta="Mean rate · 0–1">
                  <ActivityChart trace={state?.trace || []} />
                </Panel>
                <Panel title="Artificial action readout" meta="NO MOTOR OUTPUT">
                  <div className="readout">
                    {[
                      {
                        label: 'Turn',
                        value: state?.actions[0] || 0,
                        signed: true,
                      },
                      {
                        label: 'Forward',
                        value: state?.actions[1] || 0,
                        signed: false,
                      },
                    ].map((a) => (
                      <div key={a.label}>
                        <label>
                          {a.label}
                          <span className="mono">{a.value.toFixed(3)}</span>
                        </label>
                        <div className="meter">
                          <i
                            style={{
                              left: a.signed
                                ? `${Math.min(50, 50 + a.value * 50)}%`
                                : '0',
                              width: `${a.signed ? Math.abs(a.value) * 50 : a.value * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                    <p className="help">
                      Population activity → trainable readout → simulated
                      motion.
                    </p>
                  </div>
                </Panel>
              </div>
            </div>
            <aside className="side">
              <Panel
                title="Experiment controls"
                meta={<FlaskConical size={16} />}
              >
                <div className="panel-body controls">
                  <label htmlFor="condition">
                    Circuit condition
                    <Choice
                      id="condition"
                      value={state?.condition || 'intact'}
                      onChange={(condition) =>
                        command('condition', { condition })
                      }
                      items={[
                        { value: 'intact', label: 'Measured wiring' },
                        { value: 'shuffled', label: 'Shuffled signed weights' },
                        {
                          value: 'disconnected',
                          label: 'Disconnected circuit',
                        },
                      ]}
                    />
                  </label>
                  <label htmlFor="layout">
                    Neural view
                    <Choice
                      id="layout"
                      value={layout}
                      onChange={setLayout}
                      items={[
                        { value: 'layers', label: 'Circuit layers' },
                        { value: 'soma', label: 'Measured soma positions' },
                      ]}
                    />
                  </label>
                  <label htmlFor="seed">
                    Experiment seed
                    <Input
                      id="seed"
                      type="number"
                      min="0"
                      max="1000000"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </label>
                  <Button
                    variant="outline"
                    onClick={() => command('reset')}
                    disabled={!online}
                  >
                    New episode
                  </Button>
                  <label htmlFor="generations">
                    Training generations
                    <Input
                      id="generations"
                      type="number"
                      min="1"
                      max="40"
                      value={generations}
                      onChange={(e) => setGenerations(Number(e.target.value))}
                    />
                  </label>
                  <Button
                    className="full"
                    onClick={() => command('train', { generations })}
                    disabled={!online || busy}
                  >
                    <FlaskConical size={15} />
                    Train readout
                  </Button>
                  {busy && (
                    <Button
                      variant="outline"
                      onClick={() => command('cancel')}
                      disabled={state?.training.status !== 'training'}
                    >
                      <Square size={13} />
                      Cancel training
                    </Button>
                  )}
                  <div className="training-status">
                    <span className="mono">
                      {state?.training.status || 'idle'}
                    </span>
                    {state && state.training.history.length > 0 && (
                      <div>
                        Generation {state.training.history.length} · reward{' '}
                        {state.training.history.at(-1)!.reward.toFixed(1)}
                      </div>
                    )}
                  </div>
                  <p className="help">
                    Training uses measured wiring. Only the output layer learns;
                    the circuit stays fixed. Training takes place in simulation.
                  </p>
                </div>
              </Panel>
              <Panel title="Lab notebook" meta="LOCAL">
                <div className="panel-body">
                  {state?.events
                    .slice(-4)
                    .reverse()
                    .map((e, i: number) => (
                      <div className="event" key={i}>
                        <time>{e.time}</time>
                        {e.message}
                      </div>
                    ))}
                  <a
                    href={`${API}/api/export`}
                    style={{
                      display: 'inline-flex',
                      gap: 8,
                      marginTop: 16,
                      fontSize: 13,
                    }}
                  >
                    <Download size={14} />
                    Export experiment JSON
                  </a>
                </div>
              </Panel>
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="results">
          <div className="toolbar">
            <div>
              <h2 style={{ fontSize: 22, margin: '10px 0' }}>
                Measured outcomes
              </h2>
              <p className="help">
                Training reward, held-out evaluation and causal controls are
                reported separately.
              </p>
            </div>
            <Button
              variant="outline"
              disabled={!online || busy}
              onClick={() => command('compare')}
            >
              Run ablation comparison
            </Button>
          </div>
          <div className="method-grid">
            <Panel title="Training curve" meta="BEST / POPULATION MEAN">
              <ActivityChart
                training
                trace={
                  state?.training.history?.length
                    ? state.training.history
                    : state?.last_result?.history || []
                }
              />
              <div className="panel-body help">
                Seeds 11 and 23 · 12 candidate readouts / generation ·
                cross-entropy search
              </div>
            </Panel>
            <Panel title="Held-out evaluation" meta="SEEDS 101–103">
              <div className="panel-body">
                {state?.last_result ? (
                  <>
                    <table className="result-table">
                      <thead>
                        <tr>
                          <th>Readout</th>
                          <th>Reward</th>
                          <th>Cells</th>
                          <th>Contact steps</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(
                          [
                            ['Untrained', state.last_result.untrained],
                            ['Trained', state.last_result.evaluation],
                          ] as [string, Metrics][]
                        ).map(([name, r]) => (
                          <tr key={name}>
                            <td>{name}</td>
                            <td>{r.reward.toFixed(1)}</td>
                            <td>{r.coverage.toFixed(1)}</td>
                            <td>{r.collisions.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="help" style={{ marginTop: 15 }}>
                      Mean of three simulated episodes. One training seed is
                      insufficient for a general learning claim.
                    </p>
                    <div className="toolbar-group" style={{ marginTop: 14 }}>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => command('untrained')}
                      >
                        Use untrained
                      </Button>
                      <Button size="sm" onClick={() => command('trained')}>
                        Use trained
                      </Button>
                    </div>
                  </>
                ) : (
                  <p className="help">
                    Train a readout to generate held-out results.
                  </p>
                )}
              </div>
            </Panel>
          </div>
          <div className="results">
            <Panel
              title="Does the circuit matter?"
              meta={
                state?.training.status === 'comparing'
                  ? 'EVALUATING…'
                  : 'FIXED-READOUT ABLATIONS'
              }
            >
              <div className="panel-body">
                <table className="result-table">
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th>Reward</th>
                      <th>Cells explored</th>
                      <th>Contact steps</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state?.comparison.map((r) => (
                      <tr key={r.condition}>
                        <td>{r.condition}</td>
                        <td className="mono">{r.reward.toFixed(1)}</td>
                        <td className="mono">{r.coverage.toFixed(1)}</td>
                        <td className="mono">{r.collisions.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!state?.comparison.length && (
                  <p className="help" style={{ marginTop: 14 }}>
                    Run comparison to evaluate measured wiring, shuffled
                    weights, disconnection, and a direct range-based controller.
                  </p>
                )}
                <p className="help" style={{ marginTop: 18 }}>
                  These are interventions on the same readout, not equal-budget
                  retraining. A performance drop can show dependence on the
                  circuit; it does not establish biological fidelity or
                  superiority to conventional controllers.
                </p>
              </div>
            </Panel>
          </div>
        </TabsContent>
        <TabsContent value="methods">
          <div className="provenance">
            <div>
              <div className="caption">Retained neurons</div>
              <strong className="mono">
                {manifest?.nodes?.toLocaleString() || '1,252'}
              </strong>
            </div>
            <div>
              <div className="caption">Directed connections</div>
              <strong className="mono">
                {manifest?.edges?.toLocaleString() || '22,850'}
              </strong>
            </div>
            <div>
              <div className="caption">Synaptic contacts</div>
              <strong className="mono">
                {manifest?.contacts?.toLocaleString() || '163,886'}
              </strong>
            </div>
          </div>
          <div className="method-grid">
            <Panel title="Compact experiment model" meta="MODEL CARD">
              <div className="panel-body prose">
                <h2>A measured circuit, with explicit assumptions.</h2>
                <p>
                  The Experiment tab uses a cropped MaleCNS v1.0 visual network.
                  The separate Full brain tab runs the complete annotated graph
                  with approximate spiking dynamics.
                </p>
                <h3>Biological data</h3>
                <p>{manifest?.selection}</p>
                <h3>Modeled dynamics</h3>
                <p>
                  Signed leaky-rate units, normalized contact counts, tonic
                  drive and two integration updates per observation. Transmitter
                  signs are approximate. Omitted boundary inputs are not
                  restored.
                </p>
                <h3>Engineered interfaces</h3>
                <p>
                  Simulated raycast brightness drives 32 R1–R6 inputs using hex
                  positions inferred from their lamina partners. Recorded video
                  uses eight averaged luminance bins. A trained readout pools
                  visual activity into turn and forward requests.
                </p>
                <h3>What learns</h3>
                <p>
                  162 readout parameters are optimized by cross-entropy search.
                  Neural connectivity and synaptic gains remain fixed. No
                  dopamine plasticity or whole-brain learning is claimed.
                </p>
                <a
                  href="https://male-cns.janelia.org/download/"
                  target="_blank"
                  rel="noreferrer"
                >
                  MaleCNS source data · CC BY{' '}
                  <ArrowUpRight size={13} style={{ display: 'inline' }} />
                </a>
              </div>
            </Panel>
            <Panel title="From experiment to flight" meta="RESEARCH ROADMAP">
              <div className="panel-body prose">
                <ol>
                  <li>
                    <b>
                      Local circuit and reproducible benchmark — implemented.
                    </b>
                    <p>
                      Run, pause, inspect neurons, train the readout, export
                      outcomes and compare conditions.
                    </p>
                  </li>
                  <li>
                    <b>Recorded and live camera inputs — observation only.</b>
                    <p>
                      The same neural input path processes video. Proposed
                      actions cannot reach the drone from this interface.
                    </p>
                  </li>
                  <li>
                    <b>Establish learning.</b>
                    <p>
                      Train across multiple seeds and unseen layouts. Compare
                      equal training budgets, randomized circuits, direct visual
                      models, and no-vision controls. Report failures.
                    </p>
                  </li>
                  <li>
                    <b>Identify the drone dynamics.</b>
                    <p>
                      Measure camera calibration, input lag, roll/pitch response
                      and visual position estimates. Validate simulated actions
                      against this hardware.
                    </p>
                  </li>
                  <li>
                    <b>Supervised closed-loop trial.</b>
                    <p>
                      Add a separately armed bridge with command bounds, camera
                      freshness checks and independent land/stop controls. Keep
                      training out of physical flight.
                    </p>
                  </li>
                </ol>
                <p>
                  The Full brain tab now runs 166,700 annotated neurons on
                  WebGPU / Metal and shows computed spikes. Its learning and
                  camera-to-action mappings remain future work.
                </p>
              </div>
            </Panel>
            <Panel title="References and reproducibility" meta="10 SEP 2026">
              <div className="panel-body prose">
                <p>
                  <a
                    href="https://github.com/nftechie/doomfly"
                    target="_blank"
                    rel="noreferrer"
                  >
                    DOOMFLY
                  </a>{' '}
                  provides an open whole-graph example. Its published
                  experimental candidate failed learning validation gates; it is
                  useful as a methods reference.
                </p>
                <p>
                  <a
                    href="https://x.com/_lyraaaa_/status/2097527368919470162"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Beat Saber reference
                  </a>{' '}
                  and{' '}
                  <a
                    href="https://x.com/RT_Visual_on_X/status/2097310956506222660"
                    target="_blank"
                    rel="noreferrer"
                  >
                    DOOM reference
                  </a>{' '}
                  inspired the combined behavior/activity view. The clips alone
                  do not establish generalization.
                </p>
                <p>
                  Runs save seeds, model provenance, curves, held-out results
                  and readout weights locally. Times and rewards are computed
                  from the running worker.
                </p>
                {manifest?.sha256 &&
                  Object.entries(manifest.sha256).map(([k, v]) => (
                    <p className="code mono" key={k}>
                      {k} SHA256
                      <br />
                      {String(v)}
                    </p>
                  ))}
              </div>
            </Panel>
            <Panel title="Local setup" meta="NO CLOUD RUNTIME">
              <div className="panel-body prose">
                <p>Start both services from the project directory:</p>
                <pre>
                  <code>./start-lab.sh</code>
                </pre>
                <p>
                  Python runs the circuit, experiments and video decoding. The
                  browser renders activity and experiment controls. Data and
                  recordings stay on this Mac.
                </p>
                <p>
                  For live video, connect Wi-Fi to the drone. iPhone USB can
                  provide internet. This interface never changes networks or
                  sends motor commands.
                </p>
                <p>
                  Primary-camera feed only. Switch cameras separately while
                  landed, then reopen the stream.
                </p>
                <p>
                  Neuron layouts are selectable: schematic layers show inferred
                  hex placement; soma view uses only cells with published soma
                  coordinates. Connecting lines are a display subset of measured
                  edges.
                </p>
              </div>
            </Panel>
          </div>
        </TabsContent>
      </Tabs>
      <footer className="bottomnote">
        <span>
          MaleCNS v1.0 / right visual subcircuit / leaky-rate approximation
        </span>
        <span className="mono">SIMULATE → MEASURE → COMPARE → VALIDATE</span>
      </footer>
    </main>
  );
}
