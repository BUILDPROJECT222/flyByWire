'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { WORKER, WorkspaceNav, type WorkspaceState } from '../workspace-client';
type AdditionalChecks = {
  scenes: Record<
    string,
    { correct?: number; clips: number; forced_horizontal_outputs?: number }
  >;
  continuous_accuracy: number;
  latency?: {
    compute_p95_ms: number;
    receipt_p95_ms: number;
    history_p95_ms: number;
    skipped_frames: number;
    processed_frames: number;
  } | null;
};
type Diagnostics = {
  flight?: {
    phase: string;
    reason: string;
    output?: string;
    encoded_frames?: number;
    dropped_frames?: number;
    reply_age_ms?: number;
    recorder_returncode?: number;
  };
  state: WorkspaceState;
  validation: {
    summary: Record<string, { accuracy: number }>;
    clips_per_condition: number;
    phases: number;
    description: string;
    checkpoint: string;
    additional_checks?: AdditionalChecks | null;
  } | null;
};
export default function Diagnostics() {
  const [data, setData] = useState<Diagnostics | null>(null),
    [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const r = await fetch(WORKER + '/api/workspace/diagnostics');
        if (!r.ok) throw Error('Diagnostics unavailable');
        const d = (await r.json()) as Diagnostics;
        if (active) {
          setData(d);
          setError('');
        }
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) timer = setTimeout(() => void poll(), 1000);
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  const s = data?.state,
    c = s?.camera;
  const ms = (v: number | null | undefined) =>
    v == null ? '—' : v.toFixed(0) + ' ms';
  return (
    <main className="workspace-shell">
      <WorkspaceNav diagnostics />
      <div className="workspace-toolbar">
        <div>
          <h1>Diagnostics</h1>
          <p>Connection, computation and the evidence behind the model.</p>
        </div>
        <Link className="research-link" href="/experiments">
          Open research tools →
        </Link>
      </div>
      {error && (
        <p className="workspace-error" role="alert">
          {error}
        </p>
      )}
      <div className="diagnostic-grid">
        <section className="workspace-panel diagnostic-panel">
          <h2>Latest hardware review</h2>
          <p>
            12 clips · 2,849 frames · no recording drops. Eleven takeoff
            sequences were sent; one showed no clear lift, and one preparation
            expired.
          </p>
          <p>
            A reported fall after landing and a shared battery make physical
            outcomes essential. “Sequence complete” does not mean “landed
            safely.”
          </p>
          <p>
            Previous pulses at 1.2 s overlapped takeoff. New trials separate
            settling and require operator confirmation before steering.
          </p>
          <dl>
            <dt>Control gap · p95 / maximum</dt>
            <dd>55.3 / 63.1 ms</dd>
            <dt>Inference completion · p95</dt>
            <dd>72.5 ms after frame receipt</dd>
          </dl>
          <p>
            The stationary clip produced 324 leftward proposals. The brain
            remains observational; emergency cutoff was not tested in this
            batch.
          </p>
        </section>
        <section className="workspace-panel diagnostic-panel">
          <h2>Video connection</h2>
          <dl>
            <dt>Source</dt>
            <dd>{c?.source || '—'}</dd>
            <dt>Connection mode</dt>
            <dd>{c?.mode || '—'}</dd>
            <dt>Received frame age</dt>
            <dd>{ms(c?.age_ms)}</dd>
            <dt>Delivered rate</dt>
            <dd>{c?.fps?.toFixed(1) || '—'} fps</dd>
            <dt>Frame number</dt>
            <dd>{c?.sequence || 0}</dd>
          </dl>
          <p>
            {c?.last_live_error ||
              'Live RTSP is tried automatically; recording is the fallback.'}
          </p>
          <p>
            Frame age is measured from receipt on the Mac, not exposure time on
            the camera.
          </p>
        </section>
        <section className="workspace-panel diagnostic-panel">
          <h2>Full-network inference</h2>
          <dl>
            <dt>Status</dt>
            <dd>{s?.status || '—'}</dd>
            <dt>GPU</dt>
            <dd>{s?.adapter?.device || 'Loading'}</dd>
            <dt>Neurons</dt>
            <dd>{s?.manifest?.nodes.toLocaleString() || '—'}</dd>
            <dt>Connections</dt>
            <dd>{s?.manifest?.edges.toLocaleString() || '—'}</dd>
            <dt>Compute per input</dt>
            <dd>{ms(s?.compute_ms)}</dd>
            <dt>Input age at completion</dt>
            <dd>{ms(s?.source_age_ms)}</dd>
            <dt>Model time since reset</dt>
            <dd>{ms(s?.model_ms)}</dd>
            <dt>Processed frame</dt>
            <dd>{s?.inference_sequence || 0}</dd>
            <dt>Current input age</dt>
            <dd>{ms(s?.observed_input_age_ms)}</dd>
            <dt>History span</dt>
            <dd>{ms(s?.history_span_ms)}</dd>
            <dt>Frames skipped</dt>
            <dd>{s?.skipped_frames || 0}</dd>
          </dl>
          {s?.error && <p role="alert">{s.error}</p>}
        </section>
        <section className="workspace-panel diagnostic-panel">
          <h2>Motion validation</h2>
          {data?.validation ? (
            <>
              <table>
                <thead>
                  <tr>
                    <th scope="col">Frozen test</th>
                    <th scope="col">Correct</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.validation.summary).map(
                    ([key, value]) => (
                      <tr key={key}>
                        <td>
                          {{
                            intact: 'New phases',
                            shuffled: 'Shuffled frames',
                            slow_coarse: 'Slower / coarser',
                            fast_fine: 'Faster / finer',
                          }[key] || key}
                        </td>
                        <td>
                          {Math.round(
                            value.accuracy *
                              data.validation!.clips_per_condition,
                          )}{' '}
                          / {data.validation!.clips_per_condition}
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
              <p>
                {data.validation.phases} phase pairs. Synthetic gratings only;
                natural scenes and rolling camera predictions remain
                unvalidated.
              </p>
            </>
          ) : (
            <p>No saved validation report available.</p>
          )}
        </section>
        {data?.validation?.additional_checks && (
          <section className="workspace-panel diagnostic-panel">
            <h2>Beyond gratings</h2>
            <table>
              <thead>
                <tr>
                  <th scope="col">Frozen scene test</th>
                  <th scope="col">Correct</th>
                </tr>
              </thead>
              <tbody>
                {['texture', 'on_edge', 'off_edge'].map((key) => (
                  <tr key={key}>
                    <td>
                      {
                        {
                          texture: 'Moving textures',
                          on_edge: 'Light edge',
                          off_edge: 'Dark edge',
                        }[key]
                      }
                    </td>
                    <td>
                      {data.validation!.additional_checks!.scenes[key].correct}{' '}
                      / {data.validation!.additional_checks!.scenes[key].clips}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              Continuous-motion accuracy:{' '}
              {Math.round(
                data.validation.additional_checks.continuous_accuracy * 100,
              )}
              %. The binary decoder also produces a direction for stationary and
              looming inputs. It is not ready for navigation.
            </p>
            {data.validation.additional_checks.latency && (
              <>
                <h3>Recorded-video timing</h3>
                <dl>
                  <dt>Compute · p95</dt>
                  <dd>
                    {ms(
                      data.validation.additional_checks.latency.compute_p95_ms,
                    )}
                  </dd>
                  <dt>Receipt to completion · p95</dt>
                  <dd>
                    {ms(
                      data.validation.additional_checks.latency.receipt_p95_ms,
                    )}
                  </dd>
                  <dt>History span · p95</dt>
                  <dd>
                    {ms(
                      data.validation.additional_checks.latency.history_p95_ms,
                    )}
                  </dd>
                  <dt>Processed / skipped frames</dt>
                  <dd>
                    {data.validation.additional_checks.latency.processed_frames}{' '}
                    / {data.validation.additional_checks.latency.skipped_frames}
                  </dd>
                </dl>
                <p>
                  Approximately 30 seconds of replay. Excludes camera exposure,
                  radio transport and browser display latency.
                </p>
              </>
            )}
          </section>
        )}
        <section className="workspace-panel diagnostic-panel">
          <h2>Flight session</h2>
          <dl>
            <dt>Controller</dt>
            <dd>{data?.flight?.phase || 'idle'}</dd>
            <dt>TC reply age</dt>
            <dd>{ms(data?.flight?.reply_age_ms)}</dd>
            <dt>Recorded frames</dt>
            <dd>{data?.flight?.encoded_frames || 0}</dd>
            <dt>Recording drops</dt>
            <dd>{data?.flight?.dropped_frames || 0}</dd>
          </dl>
          <p>{data?.flight?.reason}</p>
          <p className="checkpoint-path">
            {data?.flight?.output || 'No flight session recorded yet.'}
          </p>
          <p>
            Each session saves flight.mp4, frame timestamps, packets, and brain
            observations. UDP delivery and physical landing are not acknowledged
            by this drone.
          </p>
        </section>
        <section className="workspace-panel diagnostic-panel">
          <h2>What is displayed</h2>
          <p>
            The camera drives the graded full-network prototype. Gold highlights
            show the strongest changes from its dark baseline, not spikes. Up to
            6,000 responses are highlighted across the anatomical point cloud;
            all neurons are computed.
          </p>
          <p>
            The readout uses 12 processed frames of T4/T5 activity. Its score is
            uncalibrated. Each frame advances 20 ms of model time, independent
            of camera time.
          </p>
          <p>
            The brain has no motor authority. The supervised test controller
            runs in a separate process. Spiking demonstrations and compact
            training experiments remain in Research tools.
          </p>
          <details>
            <summary>Checkpoint</summary>
            <p className="checkpoint-path">
              {data?.validation?.checkpoint || 'Unavailable'}
            </p>
          </details>
        </section>
      </div>
    </main>
  );
}
