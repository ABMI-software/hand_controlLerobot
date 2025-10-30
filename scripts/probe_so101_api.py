# probe_so101_api.py
from types import SimpleNamespace
from pathlib import Path
import inspect

from lerobot.robots.so101_follower.so101_follower import SO101Follower

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00"
CFG = SimpleNamespace(
    id="follower",
    port=PORT,
    calibration_dir=Path("~/.cache/huggingface/lerobot/calibration").expanduser(),
    use_degrees=True,
    disable_torque_on_disconnect=True,
    max_relative_target=15.0,
    cameras={}, polling_timeout_ms=5, connect_timeout_s=3
)

def list_candidates(obj, where):
    names = []
    for name, fn in inspect.getmembers(obj, callable):
        low = name.lower()
        # likely control methods
        if any(k in low for k in [
            "set_targets","set_positions","set_angles","set_joints",
            "set_goal","set_position","move_to","go_to","command",
            "write_angles","write_positions","set_target"
        ]):
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(?)"
            names.append((where+"."+name, sig))
    return names

r = SO101Follower(config=CFG)
print("[probe] connecting…")
r.connect()
print("[probe] connected.")

print("\n[probe] candidate methods on robot:")
for name, sig in list_candidates(r, "robot"):
    print(" -", name, sig)

if hasattr(r, "bus") and r.bus is not None:
    print("\n[probe] candidate methods on robot.bus:")
    for name, sig in list_candidates(r.bus, "bus"):
        print(" -", name, sig)

print("\n[probe] done. If no methods are listed, paste this output to me.")
