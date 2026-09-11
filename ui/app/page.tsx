'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Camera,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ArrowLeft,
  ArrowRight,
  Video,
} from 'lucide-react';
import { BrainCanvas } from './full-brain';
import { CommunitySection } from './community-section';
import { ExternalCamera } from './external-camera';
import {
  CameraPicture,
  WORKER,
  WorkspaceNav,
  useWorkspace,
} from './workspace-client';

export default function Home() {
  const { state, error } = useWorkspace();
  const [positions, setPositions] = useState<Float32Array | null>(null),
    [selected, setSelected] = useState(-1);
  const [neuron, setNeuron] = useState<{ type: string; id: string } | null>(
      null,
    ),
    [commandError, setCommandError] = useState('');
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const get = async () => {
      try {
        const r = await fetch(WORKER + '/api/full/positions');
        if (!r.ok) throw Error('Positions unavailable');
        const buffer = await r.arrayBuffer();
        if (active) setPositions(new Float32Array(buffer));
      } catch {
        if (active) timer = setTimeout(() => void get(), 3000);
      }
    };
    void get();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  async function command(action: string) {
    try {
      setCommandError('');
      const r = await fetch(WORKER + '/api/workspace/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (!r.ok) throw Error(await r.text());
    } catch (e) {
      setCommandError(String(e));
    }
  }
  async function inspect(index: number) {
    setSelected(index);
    try {
      const r = await fetch(WORKER + '/api/workspace/neuron/' + index);
      if (r.ok) setNeuron(await r.json());
    } catch {
      setNeuron(null);
    }
  }
  const camera = state?.camera,
    fresh = !!camera?.fresh && !error,
    live = camera?.source === 'live' && fresh;
  const observing =
    state?.running && state.ready && camera?.fresh && !state.error && !error;
  const motion = observing ? state.motion : null;
  return (
    <main className="workspace-shell workspace-home">
      <WorkspaceNav />
      {(error || commandError) && (
        <div className="workspace-error" role="alert">
          {commandError || error}
        </div>
      )}
      <div className="workspace-toolbar">
        <div>
          <h1>Live workspace</h1>
          <p>
            {live
              ? 'Connected to the front camera.'
              : fresh
                ? 'Playing the recorded flight. Live video connects when the drone is reachable.'
                : 'Connecting to video…'}
          </p>
        </div>
        <div className="workspace-actions">
          <Button variant="outline" onClick={() => void command('auto')}>
            <Radio size={16} />
            {camera?.mode === 'auto' ? 'Retry live' : 'Connect live'}
          </Button>
          <Button
            variant={camera?.mode === 'recording' ? 'default' : 'outline'}
            onClick={() => void command('recording')}
          >
            <Video size={16} />
            Recording
          </Button>
        </div>
      </div>
      <div className="workspace-stage">
        <section className="workspace-panel workspace-video">
          <div className="workspace-panel-heading">
            <h2>
              <Camera size={17} />
              Drone view
            </h2>
            <span className={'source-pill ' + (live ? 'is-live' : '')}>
              {live ? 'LIVE' : fresh ? 'RECORDING' : 'CONNECTING'}
            </span>
          </div>
          <div className="camera-surface">
            {fresh ? (
              <CameraPicture />
            ) : (
              <div className="camera-empty">
                <Camera size={32} />
                <p>Waiting for video</p>
                <span>
                  Connect the Mac’s Wi-Fi to the drone, or watch the demo flight replay.
                </span>
              </div>
            )}
          </div>
          <div className="workspace-panel-footer">
            <span>
              {live ? 'Front camera' : 'Recorded flight'} · 90° clockwise
            </span>
            <span>
              {camera?.mode === 'auto'
                ? 'Live connection: automatic'
                : 'Recording selected'}
            </span>
          </div>
        </section>
        <ExternalCamera />
        <section className="workspace-panel workspace-brain">
          <div className="workspace-panel-heading">
            <h2>Full brain</h2>
            <div className="workspace-actions">
              <Button
                size="sm"
                variant="ghost"
                disabled={!state?.ready}
                onClick={() => void command(state?.running ? 'pause' : 'run')}
              >
                {state?.running ? <Pause size={15} /> : <Play size={15} />}{' '}
                {state?.running ? 'Pause' : 'Run'}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={!state?.ready}
                aria-label="Reset brain response"
                onClick={() => void command('reset')}
              >
                <RotateCcw size={15} />
              </Button>
            </div>
          </div>
          <BrainCanvas
            positions={positions}
            spikes={state?.activity || []}
            selected={selected}
            onSelect={(i) => void inspect(i)}
          />
          <div className="workspace-panel-footer">
            <span>
              <i className="response-dot" />
              Response to video · continuous activity
            </span>
            <span>{state?.status || 'Loading brain'}</span>
          </div>
        </section>
      </div>
      <section className="workspace-bottom">
        <div className="motion-estimate">
          <span className="workspace-eyebrow">Unvalidated motion readout</span>
          <strong>
            {motion ? (
              <>
                {motion.direction === 'left' ? (
                  <ArrowLeft size={27} />
                ) : (
                  <ArrowRight size={27} />
                )}{' '}
                {motion.direction === 'left' ? 'Leftward' : 'Rightward'}
              </>
            ) : state?.running ? (
              'Collecting frames…'
            ) : (
              'Paused'
            )}
          </strong>
          <p>
            Stationary footage also triggered direction outputs. Brain has no
            control authority.
          </p>
        </div>
        <div className="workspace-context">
          <p>
            {state?.manifest?.nodes.toLocaleString() || '166,700'} neurons ·
            entire annotated network
          </p>
          <p>
            {neuron ? (
              <>
                <b>{neuron.type}</b> · neuron {neuron.id}
              </>
            ) : (
              'Drag to rotate. Click a neuron to inspect.'
            )}
          </p>
          <Link href="/diagnostics">View diagnostics →</Link>
        </div>
      </section>
      <CommunitySection />
      {state?.error && (
        <div role="alert" className="workspace-error">
          Video remains available. Brain: {state.error}
        </div>
      )}
    </main>
  );
}
