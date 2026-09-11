#!/usr/bin/env python3
"""One brief, supervised TC takeoff/landing trial with an onboard MP4 recording.

Default is an offline packet preview. --execute is required for all network I/O.
During execution, stdin 'land' requests landing; 'stop' cuts motors immediately.
"""
import fcntl
import argparse
import json
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HOST = "192.168.1.1"


def packet(flags=0):
    axes = [128, 128, 128, 128]
    checksum = flags
    for value in axes:
        checksum ^= value
    return bytes([3, 102, *axes, flags, checksum, 153])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        for name, flag in [("neutral", 0), ("takeoff", 1), ("land", 2), ("emergency", 4)]:
            print(name, packet(flag).hex(" "))
        return

    control_lock = (Path(__file__).resolve().parent / '.drone-control.lock').open('a')
    try:
        fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another process owns drone control; use its Land / E-stop control.')

    out = Path("recordings") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    event_file = (out / "events.jsonl").open("w", buffering=1)
    start = time.monotonic()

    def log(event, **values):
        row = {"elapsed_s": round(time.monotonic() - start, 3), "event": event, **values}
        event_file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    requests = queue.Queue()
    signal.signal(signal.SIGINT, lambda *_: requests.put("land"))
    signal.signal(signal.SIGTERM, lambda *_: requests.put("land"))

    def read_input():
        for line in sys.stdin:
            if line.strip().lower() in ("land", "stop"):
                requests.put(line.strip().lower())

    threading.Thread(target=read_input, daemon=True).start()
    recorder = None
    recorder_log = None
    airborne_possible = False
    emergency = False
    frames = {"count": 0, "last": 0.0}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("192.168.1.100", 0))
    sock.connect((HOST, 7099))
    sock.setblocking(False)
    last_rx = 0.0
    last_heartbeat = 0.0
    received_id = None
    rx_count = 0

    def pump():
        nonlocal last_rx, last_heartbeat, received_id, rx_count
        now = time.monotonic()
        if now - last_heartbeat >= 1:
            sock.send(b"\x01\x01")
            last_heartbeat = now
        for _ in range(100):
            try:
                data = sock.recv(4096)
            except BlockingIOError:
                break
            if data:
                last_rx = time.monotonic()
                received_id = data[0]
                rx_count += 1

    def send_phase(duration, flags=0):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            sock.send(packet(flags))
            try:
                pump()
            except OSError as exc:
                log("telemetry_error_during_landing", error=str(exc))
            time.sleep(0.05)

    try:
        route = subprocess.run(["/sbin/route", "-n", "get", HOST],
                               capture_output=True, text=True, timeout=3)
        if route.returncode or "interface: en0" not in route.stdout:
            raise RuntimeError("Drone route is not the expected Wi-Fi interface")
        log("preflight", local_address=sock.getsockname(), route=route.stdout)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and received_id != 83:
            pump()
            time.sleep(0.05)
        if received_id != 83:
            raise RuntimeError(f"Expected TC capability ID 83, received {received_id}")
        log("tc_capability_verified", capability_id=received_id)
        recorder_log = (out / "ffmpeg.log").open("w")
        recorder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", "udp", "-timeout", "5000000",
            "-i", f"rtsp://{HOST}:7070/webcam", "-an",
            "-vf", "transpose=clock", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+frag_keyframe+empty_moov",
            "-progress", "pipe:1", "-stats_period", "0.25", str(out / "flight.mp4")],
            stdout=subprocess.PIPE, stderr=recorder_log, text=True)

        def progress():
            for line in recorder.stdout:
                if line.startswith("frame="):
                    count = int(line.split("=", 1)[1])
                    if count > frames["count"]:
                        frames.update(count=count, last=time.monotonic())

        threading.Thread(target=progress, daemon=True).start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and frames["count"] < 20:
            pump()
            if recorder.poll() is not None:
                raise RuntimeError("Recorder exited before takeoff")
            if not requests.empty():
                raise RuntimeError("Trial cancelled before takeoff")
            time.sleep(0.05)
        if frames["count"] < 20 or time.monotonic() - last_rx > 1.5:
            raise RuntimeError("Video/telemetry not ready; no takeoff sent")
        log("recording_ready", frames=frames["count"], file=str((out / "flight.mp4").resolve()))
        flight_start = time.monotonic()
        airborne_possible = True
        log("takeoff_command", payload=packet(1).hex(), planned_land_after_s=2)
        while time.monotonic() - flight_start < 2:
            if not requests.empty():
                request = requests.get_nowait()
                emergency = request == "stop"
                log("operator_request", request=request)
                break
            now = time.monotonic()
            if recorder.poll() is not None or now - frames["last"] > 2 or now - last_rx > 1.5:
                log("early_landing", reason="video_or_telemetry_unavailable")
                break
            sock.send(packet(1 if now - flight_start < 0.15 else 0))
            pump()
            time.sleep(0.05)
    except Exception as exc:
        log("error", error=str(exc))
    finally:
        if airborne_possible:
            log("emergency_stop_command" if emergency else "landing_command",
                payload=packet(4 if emergency else 2).hex())
            landing_start = time.monotonic()
            try:
                while time.monotonic() - landing_start < (0.5 if emergency else 9):
                    if not requests.empty() and requests.get_nowait() == "stop":
                        emergency = True
                        log("operator_emergency_stop")
                        send_phase(0.5, 4)
                        break
                    elapsed = time.monotonic() - landing_start
                    # Separate TC land flag (not a takeoff/land toggle).
                    flags = 4 if emergency else (2 if elapsed < 0.25 or 3 <= elapsed < 3.25 else 0)
                    try:
                        sock.send(packet(flags))
                        pump()
                    except OSError as exc:
                        # Keep attempting landing for the bounded window even
                        # if one UDP send/receive fails.
                        log("landing_packet_error", error=str(exc))
                    time.sleep(0.05)
                sock.send(b"\x08\x01")
                log("control_session_closed", physical_landing_confirmed=False)
            except OSError as exc:
                log("landing_network_error", error=str(exc), physical_landing_confirmed=False)
        sock.close()
        if recorder is not None:
            if recorder.poll() is None:
                recorder.send_signal(signal.SIGINT)
                try:
                    recorder.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    recorder.kill()
                    recorder.wait()
            log("recording_finished", frames=frames["count"], recorder_returncode=recorder.returncode)
        if recorder_log:
            recorder_log.close()
        log("trial_finished", output=str(out.resolve()), telemetry_packets=rx_count,
            takeoff_attempted=airborne_possible)
        event_file.close()


if __name__ == "__main__":
    main()
