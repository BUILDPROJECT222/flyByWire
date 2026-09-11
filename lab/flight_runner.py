"""Supervised TC trials. A separate process owns UDP and never loads the connectome.

Optional brain-yaw-assist-v2 applies a clamped yaw bias from the workspace motion
score via shared memory. The process watches independently refreshed browser/video
leases. Record RGB frames from the workspace's sole RTSP decoder, never open a
second camera connection.
"""
import json
import hashlib
import fcntl
import multiprocessing as mp
import queue
import secrets
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HOST = '192.168.1.1'
PROFILES = {
    'hover-2': ('Take off → land · 2 s', 2., None, 0),
    'hover-4': ('Neutral hover · 4 s', 4., None, 0),
    'yaw-left': ('Small yaw − · 2 s', 2., 3, -8),
    'yaw-right': ('Small yaw + · 2 s', 2., 3, 8),
    'roll-minus': ('Small roll − · 2 s', 2., 0, -4),
    'roll-plus': ('Small roll + · 2 s', 2., 0, 4),
    'pitch-minus': ('Small pitch − · 2 s', 2., 1, -4),
    'pitch-plus': ('Small pitch + · 2 s', 2., 1, 4),
}


TRIALS = {
    'stationary-v2': ('Stationary · 10 s', 10., None, 0),
    'baseline-v2': ('Neutral baseline · 4 s', 4., None, 0),
    'yaw-minus-v2': ('Yaw −', 6., 3, -8),
    'yaw-plus-v2': ('Yaw +', 6., 3, 8),
    'roll-minus-v2': ('Roll −', 6., 0, -4),
    'roll-plus-v2': ('Roll +', 6., 0, 4),
    'pitch-minus-v2': ('Pitch −', 6., 1, -4),
    'pitch-plus-v2': ('Pitch +', 6., 1, 4),
    # Clamped yaw bias from workspace motion score; UDP process never loads the brain.
    'brain-yaw-assist-v2': ('Brain yaw assist · 8 s', 8., 3, 0),
}
# Old profiles remain only for regression tests and historical log interpretation.
PROFILES.update(TRIALS)


def packet(flags=0, axis=None, offset=0):
    axes = [128] * 4
    if axis is not None:
        axes[axis] += offset
    checksum = flags
    for value in axes:
        checksum ^= value
    return bytes([3, 102, *axes, flags, checksum, 153])


class Sequencer:
    """Clock-driven motor state machine, also used by the offline tests."""
    def __init__(self, profile='hover-2'):
        self.profile = PROFILES[profile]
        self.profile_id = profile
        self.pulse_at = None
        self.stage = 'Preflight'
        self.phase = 'preparing'
        self.since = 0.
        self.reason = 'Checking video, recording and TC replies'
        self.airborne_possible = False

    def transition(self, phase, now, reason):
        self.phase, self.since, self.reason = phase, now, reason

    def tick(self, now, *, ready=False, start=False, land=False, estop=False, fault=None, stable=False, assist_yaw=0):
        if self.phase == 'done':
            return None
        if estop and self.phase != 'estop':
            self.transition('estop', now, 'Operator E-stop: motor state is unconfirmed')
        elif self.phase not in ('landing', 'estop'):
            if land or fault:
                self.transition('landing' if self.airborne_possible else 'done', now,
                                fault or 'Operator requested landing / cancellation')
            elif self.phase == 'preparing' and ready:
                self.transition('ready', now, 'Recording ready; waiting for supervised takeoff')
            elif self.phase == 'ready' and start and ready:
                self.airborne_possible = self.profile_id != 'stationary-v2'
                self.transition('flying' if self.airborne_possible else 'recording', now, 'Scripted test; brain observes only')
            elif self.phase == 'flying' and now - self.since >= self.profile[1]:
                self.transition('landing', now, 'Scheduled landing')
        elapsed = now - self.since
        if self.phase == 'recording':
            self.stage = 'Stationary capture · no motor packets'
            if elapsed >= 10: self.transition('done', now, 'Stationary recording complete')
            return None
        if self.phase == 'flying' and self.profile_id == 'brain-yaw-assist-v2':
            if elapsed < 3:
                self.stage = 'Takeoff / settle'
                return packet(1 if elapsed < .15 else 0)
            if elapsed >= self.profile[1]:
                self.transition('landing', now, 'Assist window complete')
                return packet(2)
            self.stage = 'Brain yaw assist'
            try:
                offset = int(assist_yaw)
            except (TypeError, ValueError):
                offset = 0
            offset = max(-6, min(6, offset))
            return packet(0, 3, offset) if offset else packet(0)
        if self.phase == 'flying' and self.profile_id in TRIALS:
            self.stage = 'Takeoff / settle' if elapsed < 3 else 'Observe'
            if self.profile[2] is not None:
                if self.pulse_at is None and elapsed >= 4.5:
                    self.transition('landing', now, 'No stable-hover confirmation; pulse skipped')
                    return packet(2)
                if self.pulse_at is None and stable and 3 <= elapsed < 4.5:
                    self.pulse_at = now
                if self.pulse_at is None:
                    self.stage = 'Confirm stable hover' if elapsed >= 3 else 'Takeoff / settle'
                else:
                    since_pulse = now - self.pulse_at
                    if since_pulse >= 1.15:
                        self.transition('landing', now, 'Pulse observation complete')
                        return packet(2)
                    self.stage = 'Pulse' if since_pulse < .15 else 'Observe response'
                    if since_pulse < .15: return packet(0, self.profile[2], self.profile[3])
            return packet(1 if elapsed < .15 else 0)
        if self.phase == 'flying':
            axis, offset = (self.profile[2], self.profile[3]) if 1.2 <= elapsed < 1.35 else (None, 0)
            return packet(1 if elapsed < .15 else 0, axis, offset)
        if self.phase == 'landing':
            if elapsed >= 9:
                self.transition('done', now, 'Landing sequence sent; check motors are stopped')
                return b'\x08\x01'
            return packet(2 if elapsed < .25 or 3 <= elapsed < 3.25 else 0)
        if self.phase == 'estop':
            if elapsed >= .5:
                self.transition('done', now, 'E-stop sequence sent; check motors are stopped')
                return b'\x08\x01'
            return packet(4)
        return None


def control_process(out, profile, shared, frames, updates, address, check_route, emergency_only, lock_path=None):
    """Spawn target: bounded control loop has no numpy, GPU, camera or browser calls."""
    updates.cancel_join_thread()
    out = Path(out)
    events = (out / 'events.jsonl').open('w', buffering=1)
    log_lock = threading.Lock()
    def log(event, **data):
        with log_lock:
            events.write(json.dumps(dict(event=event, monotonic_s=time.monotonic(),
                                         utc=datetime.now(timezone.utc).isoformat(), **data)) + '\n')
    recorder = None
    recorder_thread = None
    recorder_stop = threading.Event()
    recorder_error = []
    encoded = [0, 0.]
    sock = None
    control_lock = None
    machine = Sequencer(profile)
    last_rx = last_heartbeat = 0.
    started = time.monotonic()
    prior_phase = None
    failure = None
    signal.signal(signal.SIGTERM, lambda *_: setattr(shared['land'], 'value', 1))
    signal.signal(signal.SIGINT, lambda *_: setattr(shared['land'], 'value', 1))

    def encode():
        try:
            with (out / 'frames.jsonl').open('w', buffering=1) as timestamps:
                index = 0
                while not recorder_stop.is_set() or not frames.empty():
                    try:
                        seq, received, raw = frames.get(timeout=.1)
                    except queue.Empty:
                        continue
                    remaining = memoryview(raw)
                    while remaining:
                        written = recorder.stdin.write(remaining)
                        if not written: raise IOError("Recorder pipe closed")
                        remaining = remaining[written:]
                    timestamps.write(json.dumps(dict(frame=index, pts_s=index / 10,
                        camera_sequence=seq, received_monotonic_s=received)) + '\n')
                    index += 1
                recorder.stdin.close()
        except Exception as exc:
            recorder_error.append(str(exc))

    def progress():
        for raw in recorder.stdout:
            if raw.startswith(b'frame='):
                count = int(raw.split(b'=')[1])
                if count > encoded[0]:
                    encoded[:] = [count, time.monotonic()]

    def send(payload):
        try:
            sock.send(payload)
            log('tx', hex=payload.hex(), phase=machine.phase)
            return True
        except OSError as exc:
            log('tx_error', hex=payload.hex(), error=str(exc))
            return False

    try:
        if lock_path is not None:
            control_lock = open(lock_path, 'a')
            try: fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Another process owns drone control')
        if check_route:
            route = subprocess.run(['/sbin/route', '-n', 'get', HOST], capture_output=True, text=True, timeout=3)
            log('route', output=route.stdout)
            if route.returncode or 'interface: en0' not in route.stdout:
                raise RuntimeError('Drone must be routed over Wi-Fi (en0)')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(address)
        sock.setblocking(False)
        log('session', profile=profile, local_address=sock.getsockname(), brain_controls=(profile=='brain-yaw-assist-v2'),
            rotation='90 degrees clockwise upstream', video_fps=10, emergency_only=emergency_only)
        if not emergency_only:
            recorder_log = (out / 'ffmpeg.log').open('w')
            recorder = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'warning', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '320x240', '-r', '10', '-i', 'pipe:0', '-an',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-g', '10', '-movflags', '+frag_keyframe+empty_moov', '-progress', 'pipe:1',
                '-stats_period', '0.25', str(out / 'flight.mp4')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=recorder_log, bufsize=0)
            recorder_thread = threading.Thread(target=encode, daemon=True)
            recorder_thread.start()
            threading.Thread(target=progress, daemon=True).start()
        while machine.phase != 'done':
            now = time.monotonic()
            network_fault = None
            if now - last_heartbeat >= 1:
                if not send(b'\x01\x01'):
                    network_fault = 'UDP send failed'
                last_heartbeat = now
            for _ in range(100):
                try:
                    data = sock.recv(4096)
                except BlockingIOError:
                    break
                except OSError as exc:
                    network_fault = str(exc)
                    break
                log('rx', hex=data.hex())
                # Only the observed TC status shape renews the control lease.
                if len(data) == 5 and data[0] == 83:
                    last_rx = now
            video_ok = now - shared['video'].value < 1.
            lease_ok = now - shared['lease'].value < 2.5
            recorder_ok = recorder is not None and recorder.poll() is None and not recorder_error
            ready = video_ok and lease_ok and now - last_rx < 1.5 and recorder_ok and encoded[0] >= 20 and now - encoded[1] < 1.5
            fault = network_fault
            if not emergency_only:
                if not lease_ok: fault = 'Supervising interface stopped updating'
                elif not video_ok: fault = 'Live camera stale or source changed'
                elif not recorder_ok: fault = 'Recording unavailable'
                elif shared['record_fault'].value: fault = 'Recording queue overflow'
                elif machine.phase in ('ready', 'flying', 'recording') and now - encoded[1] >= 1.5: fault = 'Recording stalled'
                elif machine.phase in ('ready', 'flying', 'recording') and now - last_rx >= 1.5: fault = 'TC replies stopped'
                elif machine.phase == 'preparing' and now - started > 15: fault = 'Preflight timed out; no takeoff sent'
                elif machine.phase == 'ready' and now - machine.since > 30: fault = 'Takeoff window expired'
            assist = float(shared['assist_yaw'].value) if 'assist_yaw' in shared else 0.
            payload = machine.tick(now, ready=ready, start=bool(shared['start'].value),
                land=bool(shared['land'].value), estop=bool(shared['estop'].value) or emergency_only, fault=fault,
                stable=bool(shared.get('stable') and shared['stable'].value), assist_yaw=assist)
            if payload is not None and not send(payload) and machine.phase == 'flying':
                machine.transition('landing', now, 'Control send failed')
            if machine.phase != prior_phase:
                log('phase', phase=machine.phase, reason=machine.reason)
                prior_phase = machine.phase
            stage = machine.stage if machine.phase in ('flying', 'recording') else machine.phase
            if stage != locals().get('last_stage'):
                log('stage', stage=stage); last_stage = stage
            updates.put(dict(phase=machine.phase, stage=stage, reason=machine.reason, encoded_frames=encoded[0],
                elapsed_s=round(now-machine.since,2), pulse_sent=machine.pulse_at is not None,
                can_confirm_stable=machine.phase=='flying' and machine.profile_id in TRIALS and machine.profile[2] is not None and machine.pulse_at is None and 3 <= now-machine.since < 4.5,
                reply_age_ms=round((now-last_rx)*1000) if last_rx else None,
                airborne_possible=machine.airborne_possible, physical_stop_confirmed=False))
            time.sleep(max(0, .05 - (time.monotonic() - now)))
    except Exception as exc:
        failure = str(exc)
        log('error', error=str(exc))
        # Unexpected software failure after takeoff still attempts a bounded landing.
        if sock is not None and machine.airborne_possible and machine.phase != 'done':
            machine.transition('landing', time.monotonic(), 'Runner error: ' + str(exc))
            while machine.phase != 'done':
                payload = machine.tick(time.monotonic(), estop=bool(shared['estop'].value))
                if payload: send(payload)
                time.sleep(.05)
        updates.put(dict(phase='error', reason=str(exc), physical_stop_confirmed=False))
    finally:
        shared['capture_done'].value = 1
        recorder_stop.set()
        if recorder:
            closing_deadline = time.monotonic() + 5
            stop_started = None
            while recorder.poll() is None or (stop_started is not None and time.monotonic() - stop_started < .5):
                now = time.monotonic()
                if shared['estop'].value and stop_started is None and machine.reason != 'E-stop sequence sent; check motors are stopped':
                    stop_started = now
                if stop_started is not None and now - stop_started < .5 and sock:
                    send(packet(4))
                if now >= closing_deadline and recorder.poll() is None: recorder.kill()
                time.sleep(.05)
            recorder.wait()
            if recorder_thread: recorder_thread.join(timeout=1)
            log('recording_finished', encoded_frames=encoded[0], returncode=recorder.returncode,
                errors=recorder_error)
            if recorder.stdin and not recorder.stdin.closed: recorder.stdin.close()
            if recorder.stdout: recorder.stdout.close()
            recorder_log.close()
        if sock and shared['estop'].value and machine.reason != 'E-stop sequence sent; check motors are stopped':
            # Cover a stop request that arrived while the encoder was finishing.
            until = time.monotonic() + .5
            while time.monotonic() < until:
                send(packet(4)); time.sleep(.05)
            send(b'\x08\x01')
            machine.reason = 'E-stop sequence sent; check motors are stopped'
        if sock: sock.close()
        result = dict(phase='error' if failure else 'done', reason=failure or machine.reason,
            encoded_frames=encoded[0], recorder_returncode=recorder.returncode if recorder else None,
            recording_errors=recorder_error, airborne_possible=machine.airborne_possible,
            physical_stop_confirmed=False, pulse_sent=machine.pulse_at is not None)
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        log('finished', **result)
        events.close()
        if control_lock: control_lock.close()


class FlightRunner:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.ctx = mp.get_context('spawn')
        self.process = None
        self.monitor = None
        self.token = None
        self.shared = None
        self.current = dict(phase='idle', reason='Prepare a supervised test', physical_stop_confirmed=False)
        previous=sorted((self.root/'recordings').glob('trial-*/metadata.json'))
        if previous:
            path=previous[-1].parent
            try:
                metadata=json.loads((path/'metadata.json').read_text())
                result=json.loads((path/'result.json').read_text())
                self.current.update(result,run_id=path.name,output=str(path),metadata=metadata)
            except (OSError,ValueError): pass

    def _drain(self):
        if self.process:
            while True:
                try: self.current.update(self.updates.get_nowait())
                except queue.Empty: break
            if not self.process.is_alive():
                result = Path(self.current['output']) / 'result.json'
                if result.exists(): self.current.update(json.loads(result.read_text()))
                elif self.current['phase'] not in ('done', 'error'):
                    self.current.update(phase='error', reason='Control process exited; motor state unknown')

    def state(self):
        with self.lock:
            self._drain()
            return dict(self.current, active=bool(self.process and self.process.is_alive()),
                        profiles=[dict(id=k, label=v[0], needs_baselines=v[2] is not None) for k,v in TRIALS.items()])

    def launch(self, workspace=None, profile='stationary-v2', emergency_only=False, metadata=None):
        with self.lock:
            if self.process and self.process.is_alive(): raise ValueError('A control session is already active')
            if self.monitor and self.monitor.is_alive(): raise ValueError('Previous recording is still closing')
            if profile not in PROFILES: raise ValueError('Unknown bounded test')
            if not emergency_only:
                if profile not in TRIALS: raise ValueError('Retired profile; choose a v2 trial')
                metadata = metadata or {}
                if not str(metadata.get('battery_id', '')).strip(): raise ValueError('Enter a battery / charge-session ID')
                if self.profile_needs_baselines(profile) and self.baseline_count(metadata['battery_id']) < 2:
                    raise ValueError('Record two clean, stable baselines before steering trials')
                status = workspace.feed.status()
                if status['source'] != 'live' or not status['fresh']: raise ValueError('Fresh live video required; replay cannot arm')
            self.shared = {key:self.ctx.Value('d', 0, lock=False) for key in ('video', 'lease', 'start', 'land', 'estop', 'record_fault', 'capture_done', 'stable', 'assist_yaw')}
            self.shared['lease'].value = time.monotonic()
            self.shared['video'].value = workspace.feed.snapshot()[4] if workspace else 0
            self.frames = self.ctx.Queue(maxsize=12)
            self.frames.cancel_join_thread()
            self.updates = self.ctx.Queue()
            self.token = secrets.token_urlsafe(24)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            out = self.root / 'recordings' / ('trial-' + stamp)
            out.mkdir(parents=True)
            metadata = dict(metadata or {}, profile=profile, protocol_version=2,
                            prepared_utc=datetime.now(timezone.utc).isoformat(), floor_level_required=True)
            (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
            self.current = dict(run_id=out.name, metadata=metadata, phase='preparing', reason='Starting independent controller', output=str(out), profile=profile,
                                physical_stop_confirmed=False, dropped_frames=0)
            self.process = self.ctx.Process(target=control_process, args=(str(out), profile, self.shared,
                self.frames, self.updates, (HOST, 7099), not emergency_only, emergency_only, str(self.root / '.drone-control.lock')))
            if workspace:
                with workspace.feed.lock: workspace.feed.flight_locked = True
            try: self.process.start()
            except Exception:
                if workspace:
                    with workspace.feed.lock: workspace.feed.flight_locked = False
                raise
            if workspace:
                self.monitor = threading.Thread(target=self._observe, args=(workspace, out), daemon=True)
                self.monitor.start()
            return dict(self.state(), token=self.token)

    def _observe(self, workspace, out):
        seq = -1
        epoch = workspace.feed.snapshot()[3]
        last_inference = -1
        try:
            with (out / 'observer.jsonl').open('w', buffering=1) as log:
                from .workspace import CHECKPOINT
                checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
                assist_on = self.current.get('profile') == 'brain-yaw-assist-v2'
                log.write(json.dumps(dict(event='model', checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_hash,
                    motor_authority=assist_on, assist='clamped yaw bias from motion score' if assist_on else None,
                    units='graded response; not spikes')) + '\n')
                from .brain_assist import score_to_yaw
                while self.process.is_alive() and not self.shared['capture_done'].value:
                    rgb, _, new_seq, new_epoch, received = workspace.feed.snapshot()
                    status = workspace.feed.status()
                    live = status['source'] == 'live' and new_epoch == epoch
                    self.shared['video'].value = received if live else 0
                    if live and rgb is not None and new_seq != seq:
                        seq = new_seq
                        try: self.frames.put_nowait((seq, received, rgb.tobytes()))
                        except queue.Full:
                            self.shared['record_fault'].value = 1
                            with self.lock: self.current['dropped_frames'] += 1
                            log.write(json.dumps(dict(event='dropped_recording_frame', sequence=seq, monotonic_s=time.monotonic())) + '\n')
                    with workspace.lock:
                        frame = {k:v for k,v in workspace.frame.items() if k != 'activity'}
                    if assist_on:
                        motion = frame.get('motion') or {}
                        yaw = score_to_yaw(motion.get('score') if isinstance(motion, dict) else None)
                        self.shared['assist_yaw'].value = float(yaw)
                    else:
                        self.shared['assist_yaw'].value = 0.
                    if frame['inference_sequence'] != last_inference:
                        last_inference = frame['inference_sequence']
                        payload = dict(event='brain_observer', monotonic_s=time.monotonic(), **frame)
                        if assist_on:
                            payload['assist_yaw'] = int(self.shared['assist_yaw'].value)
                        log.write(json.dumps(payload) + '\n')
                    time.sleep(.02)
        finally:
            self.shared['video'].value = 0
            with workspace.feed.lock: workspace.feed.flight_locked = False

    def command(self, action, token=None, confirmed=False):
        with self.lock:
            self._drain()
            active = self.process is not None and self.process.is_alive()
            if action == 'estop':
                if active: self.shared['estop'].value = 1
                else: return self.launch(emergency_only=True)
            elif action == 'land':
                if active: self.shared['land'].value = 1
                else: raise ValueError('No active control session')
            else:
                if not active or token != self.token: raise ValueError('This interface does not own the active test')
                if action == 'lease': self.shared['lease'].value = time.monotonic()
                elif action == 'stable':
                    if not self.current.get('can_confirm_stable'): raise ValueError('Stable-hover confirmation is only available after settling and before the deadline')
                    with (Path(self.current['output'])/'operator-events.jsonl').open('a') as f:
                        f.write(json.dumps(dict(event='stable_hover_confirmed',monotonic_s=time.monotonic()))+'\n')
                    self.shared['stable'].value = 1
                elif action == 'start':
                    if not confirmed: raise ValueError('Confirm upright drone, clear area and nearby supervision')
                    if self.current['phase'] != 'ready': raise ValueError('Recording and control checks are not ready')
                    self.shared['lease'].value = time.monotonic()
                    with (Path(self.current['output'])/'operator-events.jsonl').open('a') as f:
                        f.write(json.dumps(dict(event='physical_setup_confirmed', monotonic_s=time.monotonic(),
                            floor_level=True, nearby_supervision=True))+'\n')
                    self.shared['start'].value = 1
                else: raise ValueError('Unknown flight command')
            return self.state()

    @staticmethod
    def profile_needs_baselines(profile): return TRIALS[profile][2] is not None

    def baseline_count(self, battery_id):
        count=0
        for path in (self.root/'recordings').glob('trial-*/operator.json'):
            try:
                meta=json.loads((path.parent/'metadata.json').read_text())
                result=json.loads((path.parent/'result.json').read_text())
                review=json.loads(path.read_text())
                if (meta.get('profile')=='baseline-v2' and meta.get('battery_id')==battery_id
                    and review.get('outcome')=='clean_stable' and review.get('motors_stopped') is True
                    and result.get('airborne_possible') and result.get('recorder_returncode')==0
                    and result.get('reason')=='Landing sequence sent; check motors are stopped'):
                    count+=1
            except (OSError,ValueError): continue
        return count

    def save_outcome(self, run_id, outcome, motors_stopped, notes):
        with self.lock:
            if self.process and self.process.is_alive(): raise ValueError('Wait until the control session closes')
            if outcome not in ('clean_stable','drift','no_lift','incident','stationary','uncertain'):
                raise ValueError('Choose a physical outcome')
            if not run_id.startswith('trial-') or Path(run_id).name != run_id: raise ValueError('Invalid session')
            path=self.root/'recordings'/run_id
            if not (path/'result.json').exists(): raise ValueError('Completed session not found')
            review=dict(outcome=outcome,motors_stopped=motors_stopped,notes=notes,
                        source='operator report; not telemetry',utc=datetime.now(timezone.utc).isoformat())
            (path/'operator.json').write_text(json.dumps(review,indent=2))
            return review

    def close(self):
        with self.lock:
            if self.process and self.process.is_alive(): self.shared['land'].value = 1
        if self.process: self.process.join(timeout=16)
