#!/usr/bin/env python3
# Real-time absolute hand → gripper (deg-for-deg), gripper-only command.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from hand_teleop.gripper_pose.gripper_pose_computer import GripperPoseComputer
from hand_teleop.gripper_pose.gripper_pose import GripperPose

# SO-101 follower (optional)
try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
except Exception:
    SO101Follower = None

WRITE_FIELD = "Goal_Position"
DEFAULT_GRIPPER_NAME = "gripper"
DEFAULT_CALIB_PATH = Path("~/.cache/huggingface/lerobot/calibration/follower.json").expanduser()

# ------------------------------ helpers --------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))

def lerp(v: float, a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = a; b0, b1 = b
    if abs(a1 - a0) < 1e-9:
        return b0
    t = (v - a0) / (a1 - a0)
    t = min(max(t, 0.0), 1.0)
    return b0 + t * (b1 - b0)

def load_calib_range(calib_path: Path, joint_name: str) -> Optional[Tuple[float, float]]:
    try:
        data = json.loads(calib_path.read_text())
        j = data.get(joint_name)
        if not j:
            return None
        return float(j["range_min"]), float(j["range_max"])
    except Exception:
        return None

# ------------------------------ SO-101 wrapper --------------------------------
class SO101GripperOnly:
    def __init__(self, port: str, enable: bool, gripper_name: str,
                 use_degrees: bool = True,
                 calibration_dir: Optional[Path] = None,
                 max_relative_target: float = 30.0):
        self.enable = enable and (SO101Follower is not None)
        self.port = port
        self.use_degrees = use_degrees
        self.gripper_name = gripper_name
        self.calibration_dir = calibration_dir
        self.max_relative_target = max_relative_target
        self.robot: Optional[SO101Follower] = None

    def connect(self):
        if not self.enable:
            return
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            id="follower",
            port=self.port,
            calibration_dir=self.calibration_dir,
            use_degrees=self.use_degrees,
            disable_torque_on_disconnect=True,
            max_relative_target=self.max_relative_target,
            cameras={},
            polling_timeout_ms=5,
            connect_timeout_s=3,
        )
        self.robot = SO101Follower(config=cfg)
        self.robot.connect()
        try:
            self.robot.bus.configure_motors(maximum_acceleration=120, acceleration=80)
            self.robot.bus.enable_torque()
        except Exception:
            pass
        cd = "(none)" if self.calibration_dir is None else str(self.calibration_dir)
        print(f"[so101] connected on {self.port} (degrees={self.use_degrees}, calib_dir={cd})")

    def write_goal(self, value: float, normalize: bool):
        """Write ONLY the gripper joint.
        - normalize=True: value is degrees (float OK)
        - normalize=False: value is RAW counts (must be int)
        """
        if not (self.robot and self.enable):
            return
        try:
            v = int(round(value)) if not normalize else float(value)
            self.robot.bus.write(WRITE_FIELD, self.gripper_name, v, normalize=normalize)
        except Exception as e:
            print(f"[so101] write failed: {e}")

    def nudge(self, *, normalize: bool, a: float, b: float, dwell: float = 0.12):
        if not (self.robot and self.enable):
            return
        self.write_goal(a, normalize=normalize); time.sleep(dwell)
        self.write_goal(b, normalize=normalize); time.sleep(dwell)

    def close(self):
        if not (self.robot and self.enable):
            return
        try:
            self.robot.bus.disable_torque()
        except Exception:
            pass
        try:
            self.robot.disconnect()
        except Exception:
            pass
        print("[so101] disconnected")

# ----------------------------------- Main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Absolute hand→gripper (deg-for-deg), gripper-only.")
    # Video / model
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--model", type=str, default="wilor")
    ap.add_argument("--hand", type=str, default="right", choices=["left", "right"])  # right by default
    ap.add_argument("--device", type=str, default=None, help="None uses CUDA if available; or 'cpu'/'cuda:0'")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)

    # Command clamps and optional bias
    ap.add_argument("--cmd-min", type=float, default=0.0, help="Min degrees after abs()")
    ap.add_argument("--cmd-max", type=float, default=90.0, help="Max degrees after abs()")
    ap.add_argument("--offset", type=float, default=0.0, help="Added after abs(), before clamp")

    # Robot bus
    ap.add_argument("--so101-enable", action="store_true")
    ap.add_argument("--so101-port", type=str, default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00")
    ap.add_argument("--gripper-name", type=str, default=DEFAULT_GRIPPER_NAME)

    # Normalized vs RAW write
    ap.add_argument("--raw", action="store_true",
                    help="Write RAW counts (normalize=False), mapping degrees into range_min..range_max.")
    ap.add_argument("--flip-raw", action="store_true",
                    help="Flip RAW mapping direction (swap min/max).")
    ap.add_argument("--raw-min", type=float, default=None,
                    help="Override RAW range_min (counts).")
    ap.add_argument("--raw-max", type=float, default=None,
                    help="Override RAW range_max (counts).")
    ap.add_argument("--calib-path", type=str, default=str(DEFAULT_CALIB_PATH))

    # Nudge to break stiction
    ap.add_argument("--nudge", action="store_true", help="Send a tiny open/close on connect.")
    ap.add_argument("--nudge-amt", type=float, default=20.0,
                    help="Nudge amplitude: degrees in normalized mode, counts in RAW mode.")

    # Reduce bus spam / enforce delta threshold (RAW)
    ap.add_argument("--raw-delta", type=int, default=2,
                    help="Only send RAW when integer count changes by at least this many ticks.")

    # Debug
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # Pose computer (right-hand by default)
    gpc = GripperPoseComputer(device=args.device, model=args.model, hand=args.hand)

    # RAW mapping range (for --raw)
    if args.raw_min is not None and args.raw_max is not None:
        calib_range = (float(args.raw_min), float(args.raw_max))
    else:
        calib_range = load_calib_range(Path(args.calib_path), args.gripper_name)

    if args.raw and calib_range is None:
        print(f"[warn] No calibration for '{args.gripper_name}' in {args.calib_path} "
              f"and no --raw-min/--raw-max provided. Using default span 1900..2600.")
        calib_range = (1900.0, 2600.0)

    if calib_range is not None:
        rmin, rmax = calib_range
        if args.flip_raw:
            rmin, rmax = rmax, rmin
        calib_range = (rmin, rmax)
        print(f"[calib] RAW range for '{args.gripper_name}': {rmin} .. {rmax}")

    # Follower config: mirror the direct jog behavior in RAW
    if args.raw:
        follower_calib_dir = None
        max_rel = 60.0
    else:
        follower_calib_dir = DEFAULT_CALIB_PATH.parent
        max_rel = 30.0

    # Robot
    so = SO101GripperOnly(
        args.so101_port, args.so101_enable,
        gripper_name=args.gripper_name,
        use_degrees=True,
        calibration_dir=follower_calib_dir,
        max_relative_target=max_rel,
    )
    try:
        so.connect()
    except Exception as e:
        print(f"[so101] connect failed: {e}")

    # Optional nudge
    if args.nudge and so.enable and so.robot is not None:
        if args.raw and calib_range is not None:
            mid = 0.5 * (calib_range[0] + calib_range[1])
            so.nudge(normalize=False, a=mid - args.nudge_amt, b=mid + args.nudge_amt, dwell=0.15)
        else:
            so.nudge(normalize=True, a=max(0.0, 0.0), b=args.nudge_amt, dwell=0.15)

    focal_ratio = 0.9
    cam_t = np.zeros(3, dtype=np.float32)
    target_dt = 1.0 / max(1, args.fps)

    print("\nRealtime absolute control (hand → gripper):")
    if args.raw:
        print("  RAW mode: counts = lerp( clamp(|hand_deg|+offset, cmd_min, cmd_max), [cmd_min,cmd_max] → [range_min,range_max] )")
    else:
        print("  NORMALIZED mode: degrees = clamp(|hand_deg|+offset, cmd_min, cmd_max)  (normalize=True)")
    print("Press Q / ESC to quit.\n")

    last_raw_sent: Optional[int] = None

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            focal_length = focal_ratio * float(frame.shape[1])

            pose_rel: Optional[GripperPose] = gpc.compute_relative_pose(frame, focal_length, cam_t)
            if pose_rel is None:
                cv2.putText(frame, "No right hand detected...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("gripper-abs", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
                    break
                dt = time.perf_counter() - t0
                remain = target_dt - dt
                if remain > 0:
                    time.sleep(remain)
                continue

            # ---- absolute, 1:1 mapping (degrees) ----
            hand_deg_raw = float(pose_rel.open_degree)
            hand_deg_used = clamp(abs(hand_deg_raw) + float(args.offset), args.cmd_min, args.cmd_max)

            if args.raw and calib_range is not None:
                raw_target = int(round(lerp(hand_deg_used, (args.cmd_min, args.cmd_max), calib_range)))
                send = last_raw_sent is None or abs(raw_target - last_raw_sent) >= int(args.raw_delta)
                if send:
                    if args.verbose:
                        print(f"[bus] {WRITE_FIELD} {args.gripper_name} = {raw_target} (counts)  | hand_raw={hand_deg_raw:.2f}° used={hand_deg_used:.2f}°")
                    so.write_goal(raw_target, normalize=False)
                    last_raw_sent = raw_target
                shown_val = raw_target
                mode_txt = "RAW counts"
            else:
                if args.verbose:
                    print(f"[bus] {WRITE_FIELD} {args.gripper_name} = {hand_deg_used:.2f} (deg)  | hand_raw={hand_deg_raw:.2f}° used={hand_deg_used:.2f}°")
                so.write_goal(hand_deg_used, normalize=True)
                shown_val = hand_deg_used
                mode_txt = "deg (normalized)"

            # On-screen overlay
            y = 28
            cv2.putText(frame, f"hand_open_degree(raw): {hand_deg_raw:7.2f}°", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2); y += 28
            cv2.putText(frame, f"cmd ({mode_txt}): {shown_val:7.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2); y += 28
            cv2.putText(frame, f"clamp: [{args.cmd_min:.1f}, {args.cmd_max:.1f}]  offset: {args.offset:.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,255), 2)
            cv2.imshow("gripper-abs", frame)

            if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
                break

            # pacing
            dt = time.perf_counter() - t0
            remain = target_dt - dt
            if remain > 0:
                time.sleep(remain)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            so.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
