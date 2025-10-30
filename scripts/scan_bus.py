# scan_bus.py
from types import SimpleNamespace
from pathlib import Path
from lerobot.robots.so101_follower.so101_follower import SO101Follower

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00"
CFG = SimpleNamespace(
    id="scan", port=PORT,
    calibration_dir=Path("~/.cache/huggingface/lerobot/calibration").expanduser(),
    use_degrees=True, disable_torque_on_disconnect=True,
    max_relative_target=15.0, cameras={}, polling_timeout_ms=5, connect_timeout_s=3,
)

r = SO101Follower(config=CFG)
print("[scan] scanning… (this does NOT require a connect handshake)")
res = r.bus.scan_port(PORT)   # returns {baudrate: [ids]}
print("[scan] result:", res)
