"""Compare the two documented CooingDV cameras; never send flight packets."""
import json
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

out = Path("recordings") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-cameras")
out.mkdir(parents=True)
report = {"flight_commands_sent": False, "rotation": "90 degrees clockwise", "captures": []}


def capture(name, seconds):
    dest = out / (name + ".mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
               "-rtsp_transport", "udp", "-timeout", "4000000", "-i",
               "rtsp://192.168.1.1:7070/webcam", "-t", str(seconds),
               "-vf", "transpose=clock", "-an", "-c:v", "libx264", "-preset",
               "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(dest)]
    try:
        for attempt in range(3):
            result = subprocess.run(command, capture_output=True, text=True, timeout=18)
            if result.returncode == 0:
                break
            print(f"{name}: stream reopening attempt {attempt + 1} failed", flush=True)
            time.sleep(4)
        entry = {"name": name, "returncode": result.returncode, "stderr": result.stderr,
                 "video": str(dest.resolve())}
        if result.returncode == 0:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(dest),
                            "-frames:v", "1", "-update", "1", str(out / (name + ".jpg"))],
                           check=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        entry = {"name": name, "error": str(exc)}
    report["captures"].append(entry)
    print(json.dumps(entry), flush=True)


with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind(("192.168.1.100", 0))
    sock.connect(("192.168.1.1", 7099))
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                sock.send(bytes.fromhex("0101"))
            except OSError:
                return
            stop.wait(1)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        # Each capture exits and tears down RTSP before changing camera.
        capture("camera-before", 3)
        sock.send(bytes.fromhex("0602"))
        report["secondary_command"] = "0602"
        time.sleep(5)
        capture("camera-secondary", 5)
    finally:
        sock.send(bytes.fromhex("0601"))
        report["restore_command"] = "0601"
        time.sleep(5)
        capture("camera-front-restored", 2)
        stop.set()
        worker.join(timeout=2)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved camera comparison to {out.resolve()}", flush=True)
