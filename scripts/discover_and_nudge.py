# list_control_table.py
from types import SimpleNamespace
from pathlib import Path
import inspect

from lerobot.robots.so101_follower.so101_follower import SO101Follower

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00"
CALIB = Path("~/.cache/huggingface/lerobot/calibration").expanduser()

cfg = SimpleNamespace(
    id="follower", port=PORT, calibration_dir=CALIB,
    use_degrees=True, disable_torque_on_disconnect=True,
    max_relative_target=15.0, cameras={}, polling_timeout_ms=5, connect_timeout_s=3,
)

r = SO101Follower(config=cfg)
r.connect()
bus = r.bus

print("[list] looking for control-table like attributes on bus…")
for name in dir(bus):
    if any(k in name.lower() for k in ["table","reg","addr","map","name","fields","control"]):
        val = getattr(bus, name, None)
        t = type(val).__name__
        print(f"  - bus.{name} -> {t}")
        if isinstance(val, dict):
            keys = list(val.keys())
            print(f"      dict keys (first 50): {keys[:50]}")
        # some builds tuck it deeper; try to show nested dicts
        if hasattr(val, "__dict__"):
            for subn, subv in vars(val).items():
                if isinstance(subv, dict):
                    print(f"      .{subn} dict keys (first 50): {list(subv.keys())[:50]}")

print("\n[list] also scanning bus callables that mention 'write'/'read' signatures:")
for n, fn in inspect.getmembers(bus, callable):
    if n.startswith("_"): continue
    if any(k in n.lower() for k in ["write","read","position","angle","goal","target"]):
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(?)"
        print(f"  - bus.{n}{sig}")

r.disconnect()
print("[list] done.")
