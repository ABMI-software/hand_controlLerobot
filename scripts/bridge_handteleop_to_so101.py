import time
import signal
import sys
import math
import numpy as np

# ---- Hand tracking: pick ONE backend ---------------------------------
# Option A (CPU): use hand_teleop's Mediapipe wrapper (simple & robust)
# Option B (GPU): if you've installed [wilor], swap the import:
# from hand_teleop.hand_pose.estimators.wilor import WilorHandEstimator as HandEstimator

# factory utilities
from hand_teleop.hand_pose.types import Handedness

# ---- Robot (LeRobot SO-101 follower) ---------------------------------
# Import the follower directly (no CLI)
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# ---------- USER SETTINGS (edit these to your setup) -------------------
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00"  # your stable port
HANDEDNESS = "right"          # "right" or "left"
CAM_INDEX = 0                 # try 1 if wrong camera
FPS = 30
# Soft joint limits (radians) — adjust to match your calibration UI
# Order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
JOINT_MIN = np.array([-1.6, -1.2, -1.5, -1.2, -1.8, 0.0], dtype=float)
JOINT_MAX = np.array([+1.6, +1.2, +1.5, +1.2, +1.8, 1.0], dtype=float)

# Motion scaling (feel free to tune)
PAN_GAIN   = 1.2   # horizontal palm movement → shoulder_pan
LIFT_GAIN  = 1.0   # vertical palm movement   → shoulder_lift
ELB_GAIN   = 1.3   # depth (toward/away)      → elbow_flex
WRF_GAIN   = 1.0   # wrist pitch               → wrist_flex
WRR_GAIN   = 1.0   # wrist roll                → wrist_roll

# Low-pass filter (smoothing)
ALPHA = 0.35

# ---------- Helper functions ------------------------------------------
def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def lerp(a, b, t):
    return a + t * (b - a)

# ---------- Main -------------------------------------------------------
def main():
    # 1) Start hand estimator (camera)
    estimator = HandEstimator(hand=Handedness[HANDEDNESS.upper()], cam_idx=CAM_INDEX, fps=FPS)
    estimator.start()

    # 2) Connect robot
    robot = SO101Follower(
        config=dict(
            id="follower",
            port=PORT,
            # you can add: polling_timeout_ms, connect_timeout_s, etc., if needed
        )
    )
    robot.connect()
    robot.enable_torque(True)
    print("[bridge] connected to SO-101 follower on", PORT)

    # Initialize filter state at neutral
    q = np.zeros(6, dtype=float)
    q_prev = q.copy()

    # Safe neutral (center of range)
    q_home = (JOINT_MIN + JOINT_MAX) * 0.5

    # Graceful shutdown
    stop = False
    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        t_last = time.time()
        while not stop:
            # 3) Read a frame → pose estimate
            ok, pose = estimator.get_pose()  # backend returns a pose dict or None
            if not ok or pose is None:
                # If no hand detected, relax toward home softly
                q = lerp(q, q_home, 0.05)
            else:
                # --- Minimal, generic mapping ---
                # Normalized hand features you can get from the estimator:
                # pose should include: palm position (x,y,z), wrist orientation (roll,pitch), pinch/open metric, etc.
                # Different backends expose slightly different fields; adapt if field names differ.

                px = float(pose.get("palm_x", 0.0))  # left/right in meters (or normalized)
                py = float(pose.get("palm_y", 0.0))  # up/down
                pz = float(pose.get("palm_z", 0.0))  # toward camera is negative on many systems

                roll  = float(pose.get("wrist_roll", 0.0))   # radians
                pitch = float(pose.get("wrist_pitch", 0.0))  # radians
                pinch = float(pose.get("pinch", 0.0))        # 0=open…1=pinch

                # Map to SO-101 joints (heuristic but effective)
                shoulder_pan   = PAN_GAIN  * px
                shoulder_lift  = LIFT_GAIN * (-py)
                elbow_flex     = ELB_GAIN  * max(0.0, -pz)   # bring elbow in as hand comes closer
                wrist_flex     = WRF_GAIN  * pitch
                wrist_roll     = WRR_GAIN  * roll
                gripper        = pinch

                target = np.array([shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], dtype=float)

                # Clip to safe limits
                q = clip(target, JOINT_MIN, JOINT_MAX)

                # Smooth
                q = ALPHA * q + (1.0 - ALPHA) * q_prev

            # 4) Send to robot
            robot.set_targets(q.tolist())

            # 5) Housekeeping
            q_prev = q.copy()
            # Run near-FPS
            dt = max(0.0, (1.0 / FPS) - (time.time() - t_last))
            if dt > 0:
                time.sleep(dt)
            t_last = time.time()

    finally:
        # On exit: go to a soft safe pose, then torque off
        try:
            for _ in range(30):
                q_prev = lerp(q_prev, q_home, 0.2)
                robot.set_targets(q_prev.tolist())
                time.sleep(1/50)
        except Exception:
            pass
        robot.enable_torque(False)
        robot.disconnect()
        estimator.stop()
        print("\n[bridge] stopped cleanly")

if __name__ == "__main__":
    main()
