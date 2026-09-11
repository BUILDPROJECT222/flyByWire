'use client';
/* Live JPEG frames must bypass the image optimizer. */
/* oxlint-disable next/no-img-element */
import Link from 'next/link';
import { FlightControls } from './flight-controls';
import { useEffect, useState } from 'react';
export const WORKER = 'http://127.0.0.1:8766';
export type WorkspaceState = {
  ready: boolean;
  running: boolean;
  status: string;
  error: string | null;
  activity: number[][];
  motion: { direction: string; score: number; frames: number } | null;
  model_ms: number;
  compute_ms: number;
  inference_sequence: number;
  source_age_ms: number | null;
  hardware_enabled: boolean;
  result_age_ms?: number | null;
  observed_input_age_ms?: number | null;
  processed_frames?: number;
  skipped_frames?: number;
  history_span_ms?: number | null;
  camera: {
    mode: string;
    source: string;
    detail: string;
    sequence: number;
    age_ms: number | null;
    fresh: boolean;
    fps: number;
    last_live_error: string | null;
  };
  manifest?: { nodes: number; edges: number; positions: number };
  adapter?: { device: string; backend_type: string };
};
export function useWorkspace() {
  const [state, setState] = useState<WorkspaceState | null>(null),
    [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const r = await fetch(WORKER + '/api/workspace');
        if (!r.ok) throw Error('Local worker unavailable');
        const value = (await r.json()) as WorkspaceState;
        if (active) {
          setState(value);
          setError('');
        }
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) timer = setTimeout(() => void poll(), 400);
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  return { state, error };
}
export function WorkspaceNav({
  diagnostics = false,
}: {
  diagnostics?: boolean;
}) {
  return (
    <>
      <header className="workspace-header">
        <Link className="workspace-brand" href="/">
          fly<span className="brand-by">By</span>Wire
        </Link>
        <nav aria-label="Main navigation">
          <Link href="/" aria-current={!diagnostics ? 'page' : undefined}>
            Workspace
          </Link>
          <Link
            href="/diagnostics"
            aria-current={diagnostics ? 'page' : undefined}
          >
            Diagnostics
          </Link>
        </nav>
        <span className="workspace-observe">Brain: observation only</span>
      </header>
      <FlightControls />
    </>
  );
}
export function CameraPicture() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick(Date.now()), 100);
    return () => clearInterval(timer);
  }, []);
  return (
    <img
      className="camera-picture"
      src={WORKER + '/api/workspace/frame.jpg?t=' + tick}
      alt="Current drone camera frame"
    />
  );
}
