"""Send three documented CooingDV heartbeats; no flight/control frames."""
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

result = {"time": datetime.now(timezone.utc).isoformat(),
          "destination": ["192.168.1.1", 7099],
          "transmitted_payload_hex": "0101", "flight_commands_sent": False,
          "replies": []}
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind(("192.168.1.100", 0))
    sock.connect(("192.168.1.1", 7099))
    result["local_address"] = sock.getsockname()
    for _ in range(3):
        sock.send(bytes.fromhex("0101"))
        end = time.monotonic() + 1
        while time.monotonic() < end:
            sock.settimeout(max(0.001, end - time.monotonic()))
            try:
                data = sock.recv(4096)
                result["replies"].append({"hex": data.hex(), "length": len(data)})
            except socket.timeout:
                break
            except OSError as exc:
                result["error"] = str(exc)
                break
path = Path("diagnostics") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-telemetry.json")
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
print(path.resolve())
