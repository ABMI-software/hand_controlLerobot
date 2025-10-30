#!/usr/bin/env python3
from types import SimpleNamespace
import time
try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
except Exception as e:
    raise SystemExit(f"Import error: {e}")

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00"
JOINT = "gripper"
# Try a wide, safe span. Adjust if needed.
P0, P1, P2 = 2000, 2400, 2600   # raw counts

cfg = SimpleNamespace(
    id="follower",
    port=PORT,
    calibration_dir=None,
    use_degrees=True,  # doesn't matter for RAW when normalize=False
    disable_torque_on_disconnect=True,
    max_relative_target=60.0,
    cameras={},
    polling_timeout_ms=5,
    connect_timeout_s=3,
)
bot = SO101Follower(config=cfg); bot.connect()
try:
    try:
        bot.bus.configure_motors(maximum_acceleration=120, acceleration=80)
        bot.bus.enable_torque()
    except Exception:
        pass

    for target in (P0, P1, P2, P0):
        print(f"Writing {target} counts RAW to {JOINT}")
        bot.bus.write("Goal_Position", JOINT, int(target), normalize=False)
        time.sleep(0.6)

finally:
    try:
        bot.bus.disable_torque()
    except Exception:
        pass
    bot.disconnect()
    print("Done.")
