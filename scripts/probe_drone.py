#!/usr/bin/env python3
"""Bounded camera/network diagnostic; never sends flight-control commands."""
import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        return {"command": args, "error": f"Timed out after {timeout}s"}
    except OSError as exc:
        return {"command": args, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.1")
    parser.add_argument("--output", default="diagnostics")
    args = parser.parse_args()
    socket.inet_aton(args.host)
    out = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    report = {"host": args.host, "flight_commands_sent": False}
    report["route"] = run(["/sbin/route", "-n", "get", args.host])
    print(json.dumps(report["route"], indent=2), flush=True)
    report["ping"] = run(["/sbin/ping", "-c", "3", "-W", "1000", args.host], 6)
    try:
        with socket.create_connection((args.host, 7070), timeout=3) as sock:
            report["rtsp_port_open"] = True
            sock.settimeout(3)
            sock.sendall((f"OPTIONS rtsp://{args.host}:7070/webcam RTSP/1.0\r\n"
                          "CSeq: 1\r\nUser-Agent: flyByWire-probe\r\n\r\n").encode())
            report["rtsp_options"] = sock.recv(8192).decode(errors="replace")
    except OSError as exc:
        report["rtsp_error"] = str(exc)
    if report.get("rtsp_port_open"):
        url = f"rtsp://{args.host}:7070/webcam"
        for transport in ("tcp", "udp"):
            prefix = ["-rtsp_transport", transport, "-timeout", "5000000"]
            report[f"video_{transport}"] = run([
                "ffprobe", "-v", "error", *prefix, "-show_streams", "-of", "json", url], 12)
            probe = report[f"video_{transport}"]
            if probe.get("returncode") == 0 and '"codec_type": "video"' in probe["stdout"]:
                report["snapshot"] = run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", *prefix,
                    "-i", url, "-frames:v", "1", "-update", "1", str(out / "camera.jpg")], 12)
                if report["snapshot"].get("returncode") == 0:
                    break
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved diagnostics to {out.resolve()}")


if __name__ == "__main__":
    main()
