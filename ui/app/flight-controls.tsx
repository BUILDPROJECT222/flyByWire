'use client';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';

import { WORKER as API } from '@/lib/worker';
type Flight = {
  phase: string;
  reason: string;
  active: boolean;
  output?: string;
  encoded_frames?: number;
  dropped_frames?: number;
  run_id?: string;
  profile?: string;
  stage?: string;
  can_confirm_stable?: boolean;
  profiles: { id: string; label: string; needs_baselines?: boolean }[];
};

export function FlightControls() {
  const [state, setState] = useState<Flight | null>(null);
  const [profile, setProfile] = useState('stationary-v2');
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const battery = 'same-battery';
  const [baselines, setBaselines] = useState(0);
  const [outcome, setOutcome] = useState('uncertain');
  const [stopped, setStopped] = useState(false);
  const [notes, setNotes] = useState('');
  const [saved, setSaved] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);
  useEffect(() => {
    let active = true;
    void fetch(API + '/api/flight/baselines?battery_id=same-battery')
      .then((r) => r.json())
      .then((value) => {
        if (active) setBaselines((value as { count: number }).count);
      })
      .catch(() => {
        if (active) setBaselines(0);
      });
    return () => {
      active = false;
    };
  }, [saved]);
  const token = useRef<string | null>(null);
  useEffect(() => {
    token.current = sessionStorage.getItem('flight-owner');
    let mounted = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (token.current)
          await fetch(API + '/api/flight/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'lease', token: token.current }),
            signal: AbortSignal.timeout(900),
          });
        const r = await fetch(API + '/api/flight', {
          signal: AbortSignal.timeout(900),
        });
        if (!r.ok) throw Error('Controller unavailable');
        const next = (await r.json()) as Flight;
        if (mounted) {
          setState(next);
          if (next.active && next.profile) setProfile(next.profile);
          if (!next.active) {
            token.current = null;
            sessionStorage.removeItem('flight-owner');
          }
        }
      } catch {
        if (mounted) setState(null);
      } finally {
        if (mounted) timer = setTimeout(() => void poll(), 500);
      }
    };
    void poll();
    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, []);
  async function command(action: string) {
    if (action !== 'estop' && action !== 'land') setBusy(true);
    setError('');
    setNotice(action === 'estop' ? 'Requesting E-stop…' : '');
    try {
      const r = await fetch(API + '/api/flight/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          profile,
          token: token.current,
          confirmed,
          battery_id: battery,
          charge: 'unknown',
          run_id: state?.run_id,
          outcome,
          motors_stopped: stopped,
          notes,
        }),
        signal: AbortSignal.timeout(2000),
      });
      const next = (await r.json()) as Flight & {
        token?: string;
        detail?: string;
      };
      if (!r.ok) throw Error(next.detail || 'Request failed');
      if (action === 'outcome') {
        setSaved(true);
        setReviewOpen(false);
        setNotice('Outcome saved.');
        return;
      }
      if (next.token) {
        token.current = next.token;
        sessionStorage.setItem('flight-owner', next.token);
      }
      setState(next);
      if (action === 'prepare') {
        setSaved(false);
        setStopped(false);
        setOutcome('uncertain');
        setReviewOpen(false);
      }
      if (action === 'prepare' || action === 'start') setConfirmed(false);
      if (action === 'estop')
        setNotice('E-stop requested; check motors. Delivery is unconfirmed.');
      if (action === 'land')
        setNotice('Landing requested; check the physical drone.');
    } catch (e) {
      setError(
        (action === 'estop' ? 'E-stop could not be confirmed. ' : '') +
          String(e),
      );
      setNotice('');
    } finally {
      setBusy(false);
    }
  }
  const active = !!state?.active;
  const needsBaselines = !!state?.profiles.find((p) => p.id === profile)
    ?.needs_baselines;
  const primaryAction = state?.can_confirm_stable
    ? 'stable'
    : state?.phase === 'ready'
      ? 'start'
      : 'prepare';
  const primaryLabel = state?.can_confirm_stable
    ? 'Stable — apply pulse'
    : state?.phase === 'ready'
      ? profile === 'stationary-v2'
        ? 'Record 10 s'
        : 'Take off'
      : active
        ? state?.phase === 'preparing'
          ? 'Preparing…'
          : 'Running…'
        : 'Prepare';
  const primaryDisabled =
    busy ||
    !state ||
    (primaryAction === 'prepare' &&
      (active || (needsBaselines && baselines < 2))) ||
    (primaryAction === 'start' && !confirmed);
  return (
    <section
      className="flight-controls flight-compact"
      aria-label="Supervised flight controls"
    >
      <div className="flight-buttons">
        <label htmlFor="flight-profile" className="flight-label">
          Trial
        </label>
        <select
          id="flight-profile"
          value={profile}
          disabled={active || busy}
          onChange={(e) => setProfile(e.target.value)}
        >
          {(
            state?.profiles || [
              {
                id: 'stationary-v2',
                label: 'Stationary · 10 s',
                needs_baselines: false,
              },
            ]
          ).map((p) => (
            <option
              key={p.id}
              value={p.id}
              disabled={p.needs_baselines && baselines < 2}
            >
              {p.label}
            </option>
          ))}
        </select>
        <Button
          disabled={primaryDisabled}
          onClick={() => void command(primaryAction)}
        >
          {primaryLabel}
        </Button>
        <Button
          variant="outline"
          disabled={state !== null && !active}
          onClick={() => void command('land')}
        >
          Land / cancel
        </Button>
        <button
          className="flight-estop"
          type="button"
          onClick={() => void command('estop')}
          aria-label="Emergency stop: cut drone motors"
        >
          E-STOP <span>Cut motors</span>
        </button>
      </div>
      <div className="flight-compact-status">
        <span aria-live="polite">
          {state?.stage || state?.phase || 'Unavailable'} ·{' '}
          {state?.reason || 'Local controller unavailable'}
        </span>
        {!active && (
          <span className="flight-baseline-count">
            {Math.min(2, baselines)}/2 clean baselines
          </span>
        )}
        {!active && state?.run_id && (
          <button
            className="flight-text-button"
            onClick={() => setReviewOpen(!reviewOpen)}
          >
            Outcome{saved ? ' ✓' : ''}
          </button>
        )}
      </div>
      {state?.phase === 'ready' && (
        <label className="flight-confirm">
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(e) => setConfirmed(e.target.checked)}
          />
          Motors stopped, drone upright on the floor, area clear; I am beside
          it.
        </label>
      )}
      {(error || notice) && (
        <p className={error ? 'flight-error' : 'flight-notice'} role="alert">
          {error || notice}
        </p>
      )}
      {reviewOpen && !active && state?.run_id && (
        <div
          className="flight-review-popover"
          role="dialog"
          aria-label="Physical trial outcome"
        >
          <div className="flight-review-title">
            <strong>What happened?</strong>
            <button
              className="flight-text-button"
              onClick={() => setReviewOpen(false)}
            >
              Close
            </button>
          </div>
          <label>
            Outcome
            <select
              value={outcome}
              onChange={(e) => setOutcome(e.target.value)}
            >
              <option value="uncertain">Uncertain</option>
              <option value="stationary">Stationary recording</option>
              <option value="clean_stable">Stable hover + clean landing</option>
              <option value="drift">Lifted, with drift</option>
              <option value="no_lift">Did not lift</option>
              <option value="incident">
                Fall / collision / unstable landing
              </option>
            </select>
          </label>
          <label className="flight-confirm">
            <input
              type="checkbox"
              checked={stopped}
              onChange={(e) => setStopped(e.target.checked)}
            />{' '}
            Motors stopped (observed)
          </label>
          <label>
            Notes (optional)
            <input
              value={notes}
              maxLength={1000}
              onChange={(e) => setNotes(e.target.value)}
            />
          </label>
          <Button onClick={() => void command('outcome')} disabled={busy}>
            Save outcome
          </Button>
        </div>
      )}
    </section>
  );
}
