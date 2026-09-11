"""Offline flight validation: all UDP destinations are loopback fake drones."""
import json
import fcntl
import multiprocessing as mp
import queue
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from .flight_runner import FlightRunner, Sequencer, packet, control_process, PROFILES, TRIALS


class SequencerTests(unittest.TestCase):
    def flying(self, profile='hover-2'):
        s = Sequencer(profile)
        self.assertIsNone(s.tick(0, start=True))
        s.tick(1, ready=True)
        self.assertEqual(s.phase, 'ready')
        self.assertEqual(s.tick(2, ready=True, start=True), packet(1))
        return s

    def test_verified_packets(self):
        self.assertEqual(packet(1).hex(), '036680808080010199')
        self.assertEqual(packet(2).hex(), '036680808080020299')
        self.assertEqual(packet(4).hex(), '036680808080040499')

    def test_default_sequence_and_repeat_land(self):
        s = self.flying()
        self.assertEqual(s.tick(2.1), packet(1))
        self.assertEqual(s.tick(2.2), packet())
        self.assertEqual(s.tick(4), packet(2))
        self.assertEqual(s.tick(4.3), packet())
        self.assertEqual(s.tick(7.1), packet(2))
        self.assertEqual(s.tick(13), b'\x08\x01')
        self.assertFalse(s.tick(14, start=True, ready=True))

    def test_estop_overrides_every_phase_and_start(self):
        for phase in ('preparing', 'ready', 'flying', 'landing'):
            s = Sequencer(); s.phase = phase
            self.assertEqual(s.tick(10, ready=True, start=True, land=True, estop=True), packet(4))
            self.assertEqual(s.tick(10.25, ready=True, start=True), packet(4))
            self.assertEqual(s.tick(10.51, ready=True, start=True), b'\x08\x01')
            self.assertIsNone(s.tick(11, start=True, ready=True))

    def test_fault_cancels_preflight_and_lands_after_takeoff(self):
        s = Sequencer(); s.tick(0, fault='stale video', ready=True, start=True)
        self.assertEqual(s.phase, 'done')
        self.assertFalse(s.airborne_possible)
        s = self.flying()
        self.assertEqual(s.tick(2.3, fault='stale video'), packet(2))
        self.assertEqual(s.phase, 'landing')

    def test_all_pulses_are_bounded_and_throttle_remains_centered(self):
        for profile, (_, duration, axis, offset) in PROFILES.items():
            if profile in TRIALS: continue
            s = self.flying(profile)
            before, during, after = s.tick(3.19), s.tick(3.25), s.tick(3.36)
            self.assertEqual(before, packet())
            self.assertEqual(during, packet(0, axis, offset))
            self.assertEqual(after, packet())
            self.assertEqual(during[4], 128)
            self.assertEqual(s.tick(2 + duration), packet(2))

    def test_replay_cannot_launch_and_start_requires_owner(self):
        class Feed:
            def status(self): return dict(source='recording', fresh=True)
        class Workspace: feed = Feed()
        with tempfile.TemporaryDirectory() as tmp:
            f = FlightRunner(tmp)
            with self.assertRaisesRegex(ValueError, 'replay cannot arm'): f.launch(Workspace(),metadata={'battery_id':'test'})
            self.assertIsNone(f.process)
            with self.assertRaises(ValueError): f.command('start', confirmed=True)

    def test_confirmation_and_owner_are_required_and_estop_is_unrestricted(self):
        f = FlightRunner('/tmp')
        f.process = SimpleNamespace(is_alive=lambda: True)
        f.updates = queue.Queue()
        f.shared = {k:SimpleNamespace(value=0) for k in ('start','lease','estop','land')}
        f.current['phase'] = 'ready'; f.token = 'owner'
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); f.current['output']=temp.name
        with self.assertRaises(ValueError): f.command('start', 'wrong', True)
        with self.assertRaises(ValueError): f.command('start', 'owner', False)
        self.assertEqual(f.shared['start'].value, 0)
        f.command('start', 'owner', True)
        self.assertEqual(f.shared['start'].value, 1)
        f.command('estop')
        self.assertEqual(f.shared['estop'].value, 1)


class LoopbackTests(unittest.TestCase):
    def run_trial(self, scenario):
        ctx = mp.get_context('spawn')
        shared = {k:ctx.Value('d', 0, lock=False) for k in
                  ('video','lease','start','land','estop','record_fault','capture_done','stable')}
        shared['video'].value = shared['lease'].value = time.monotonic()
        frames, updates = ctx.Queue(maxsize=12), ctx.Queue()
        frames.cancel_join_thread()
        stop = threading.Event()
        packets = []
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as drone:
            drone.bind(('127.0.0.1', 0)); drone.settimeout(.05)
            def receive():
                while not stop.is_set():
                    try: data, peer = drone.recvfrom(100)
                    except socket.timeout: continue
                    packets.append((time.monotonic(), data))
                    if data == b'\x01\x01': drone.sendto(b'\x53\x01\x00\x00\x00', peer)
            receiver = threading.Thread(target=receive); receiver.start()
            p = ctx.Process(target=control_process, args=(tmp, 'yaw-minus-v2' if scenario == 'v2-pulse' else 'hover-4' if scenario == 'browser-fault' else 'hover-2', shared, frames, updates,
                                                         drone.getsockname(), False, False))
            p.start()
            started = time.monotonic(); seq = 0; last_frame = 0.; flying_at = None; landing_at = None
            phase = 'preparing'
            try:
                while p.is_alive() and time.monotonic() - started < 22:
                    now = time.monotonic()
                    while True:
                        try: phase = updates.get_nowait()['phase']
                        except queue.Empty: break
                    if phase == 'ready': shared['start'].value = 1
                    if phase == 'flying' and flying_at is None: flying_at = now
                    if phase == 'landing' and landing_at is None: landing_at = now
                    if scenario == 'v2-pulse' and flying_at and now-flying_at>=3.2: shared['stable'].value=1
                    if not (scenario == 'browser-fault' and flying_at and now - flying_at > .2):
                        shared['lease'].value = now
                    if scenario == 'video-fault-estop' and flying_at and now - flying_at > .2:
                        shared['video'].value = 0
                    elif now - last_frame >= .1 and not shared['capture_done'].value:
                        # Known RGB frame, already in dashboard orientation.
                        seq += 1; last_frame = now; shared['video'].value = now
                        frames.put_nowait((seq, now, bytes([80, 140, 200]) * (320*240)))
                    if scenario in ('video-fault-estop', 'browser-fault') and landing_at and now - landing_at > .15:
                        shared['estop'].value = 1
                    time.sleep(.01)
                p.join(1)
                self.assertFalse(p.is_alive(), 'Control process exceeded its bounded trial')
                result = json.loads((Path(tmp)/'result.json').read_text())
                self.assertEqual(result['phase'], 'done', result)
                self.assertEqual(result['recorder_returncode'], 0, (result, (Path(tmp)/'ffmpeg.log').read_text()))
                self.assertFalse(result['physical_stop_confirmed'])
                controls = [(t,x) for t,x in packets if len(x)==9]
                flags = [x[6] for _,x in controls]
                self.assertIn(1, flags); self.assertIn(2, flags)
                if scenario=='v2-pulse':
                    pulse=[(t,x) for t,x in controls if x[2:6] != bytes([128]*4)]
                    self.assertGreaterEqual(len(pulse),2)
                    self.assertGreater(pulse[0][0]-controls[0][0],3.)
                    self.assertLess(pulse[-1][0]-pulse[0][0],.2)
                    self.assertTrue(all(x==packet(0,3,-8) for _,x in pulse))
                self.assertEqual(packets[-1][1], b'\x08\x01')
                if scenario in ('video-fault-estop', 'browser-fault'):
                    self.assertIn(4, flags)
                    first_stop = flags.index(4)
                    self.assertTrue(all(f == 4 for f in flags[first_stop:]))
                    self.assertLess(landing_at - flying_at, 3.2 if scenario == 'browser-fault' else 1.)
                    if scenario == 'browser-fault':
                        self.assertGreater(landing_at - flying_at, 2.4)
                        self.assertIn('Supervising interface stopped updating', (Path(tmp)/'events.jsonl').read_text())
                else:
                    self.assertNotIn(4, flags)
                    land_times = [t for t,x in controls if x[6]==2]
                    self.assertGreater(land_times[-1] - land_times[0], 3.)
                    times = [t for t,_ in controls]
                    gaps = sorted(b-a for a,b in zip(times,times[1:]))
                    self.assertLess(gaps[int(.95*len(gaps))], .1)
                probe = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=width,height,nb_read_frames','-of','json',str(Path(tmp)/'flight.mp4')],
                    capture_output=True, text=True, check=True)
                video = json.loads(probe.stdout)['streams'][0]
                self.assertEqual((video['width'], video['height']), (320, 240))
                self.assertGreaterEqual(int(video['nb_read_frames']), 20)
                timestamps = (Path(tmp)/'frames.jsonl').read_text().splitlines()
                self.assertEqual(len(timestamps), int(video['nb_read_frames']))
            finally:
                stop.set(); receiver.join(1)
                if p.is_alive(): p.kill(); p.join()
                frames.close(); updates.close()

    def test_complete_recorded_hover_and_landing(self): self.run_trial('normal')
    def test_video_loss_lands_and_estop_interrupts_landing(self): self.run_trial('video-fault-estop')
    def test_browser_lease_loss_lands_independently(self): self.run_trial('browser-fault')
    def test_v2_confirmed_pulse_records_and_lands(self): self.run_trial('v2-pulse')


class EmergencyProcessTests(unittest.TestCase):
    def run_stop_only(self, locked=False):
        ctx = mp.get_context('spawn')
        shared = {k:ctx.Value('d', 0, lock=False) for k in
                  ('video','lease','start','land','estop','record_fault','capture_done','stable')}
        shared['start'].value = 1  # Emergency-only must ignore even a pending start.
        frames, updates = ctx.Queue(), ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as drone:
            drone.bind(('127.0.0.1', 0)); drone.settimeout(.05)
            lock_path = str(Path(tmp)/'controller.lock')
            owner = open(lock_path, 'a')
            if locked: fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            p = ctx.Process(target=control_process, args=(tmp,'hover-2',shared,frames,updates,
                drone.getsockname(),False,True,lock_path))
            p.start(); packets=[]; deadline=time.monotonic()+3
            try:
                while time.monotonic()<deadline:
                    try: packets.append(drone.recv(100))
                    except socket.timeout:
                        if not p.is_alive(): break
                p.join(1)
                self.assertFalse(p.is_alive())
                result=json.loads((Path(tmp)/'result.json').read_text())
                self.assertFalse(result['airborne_possible'])
                self.assertFalse((Path(tmp)/'flight.mp4').exists())
                if locked:
                    self.assertEqual(packets, [])
                    self.assertEqual(result['phase'],'error')
                    self.assertIn('Another process owns',result['reason'])
                else:
                    controls=[p for p in packets if len(p)==9]
                    self.assertGreaterEqual(len(controls),8)
                    self.assertTrue(all(p==packet(4) for p in controls))
                    self.assertEqual(packets[-1],b'\x08\x01')
            finally:
                if p.is_alive(): p.kill(); p.join()
                owner.close(); frames.close(); updates.close()

    def test_estop_without_camera_or_prepared_session(self): self.run_stop_only()
    def test_competing_controller_cannot_send(self): self.run_stop_only(locked=True)


class RevisedTrialTests(unittest.TestCase):
    def start(self, profile):
        s=Sequencer(profile); s.tick(0,ready=True); s.tick(1,ready=True,start=True)
        return s

    def test_stationary_never_sends_motor_commands(self):
        s=self.start('stationary-v2')
        self.assertFalse(s.airborne_possible)
        for now in (1.1,2,5,10.9,11,12): self.assertIsNone(s.tick(now,stable=True))
        self.assertEqual(s.phase,'done')

    def test_no_early_pulse_and_no_confirmation_means_land(self):
        s=self.start('yaw-minus-v2')
        for now in (1.2,2.2,3.9): self.assertEqual(s.tick(now),packet())
        self.assertEqual(s.tick(4),packet())
        self.assertEqual(s.stage,'Confirm stable hover')
        self.assertEqual(s.tick(5.5),packet(2))
        self.assertIsNone(s.pulse_at)

    def test_every_axis_has_one_bounded_confirmed_pulse(self):
        for profile,(_,_,axis,offset) in TRIALS.items():
            if axis is None: continue
            s=self.start(profile)
            self.assertEqual(s.tick(2.3),packet())
            self.assertEqual(s.tick(4.2,stable=True),packet(0,axis,offset))
            self.assertEqual(s.tick(4.3,stable=True),packet(0,axis,offset))
            self.assertEqual(s.tick(4.4,stable=True),packet())
            self.assertEqual(s.tick(5.36,stable=True),packet(2))
            self.assertEqual(s.phase,'landing')

    def test_fault_and_estop_override_confirmed_pulse(self):
        for kwargs,expected in [({'fault':'video lost'},packet(2)),({'estop':True},packet(4))]:
            s=self.start('yaw-plus-v2')
            self.assertEqual(s.tick(4.1,stable=True,**kwargs),expected)
            self.assertIsNone(s.pulse_at)

    def test_hard_deadline_does_not_extend_on_late_confirmation(self):
        s=self.start('roll-plus-v2')
        self.assertEqual(s.tick(7,stable=True),packet(2))
        self.assertIsNone(s.pulse_at)

    def test_baseline_unlock_uses_matching_battery_and_physical_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=FlightRunner(tmp)
            for i in range(2):
                p=Path(tmp)/'recordings'/f'trial-{i}';p.mkdir(parents=True)
                (p/'metadata.json').write_text(json.dumps(dict(profile='baseline-v2',battery_id='A-1')))
                (p/'result.json').write_text(json.dumps(dict(airborne_possible=True,recorder_returncode=0,
                    reason='Landing sequence sent; check motors are stopped')))
                f.save_outcome(p.name,'clean_stable',True,'Observed from beside the drone')
            self.assertEqual(f.baseline_count('A-1'),2)
            self.assertEqual(f.baseline_count('A-2'),0)
            f.save_outcome('trial-1','incident',True,'Drifted after landing')
            self.assertEqual(f.baseline_count('A-1'),1)
            with self.assertRaises(ValueError):f.save_outcome('../outside','incident',True,'')


if __name__ == '__main__': unittest.main()
