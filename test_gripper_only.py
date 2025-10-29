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

try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
except Exception:
    SO101Follower = None

WRITE_FIELD = "Goal_Position"
DEFAULT_GRIPPER_NAME = "gripper"
DEFAULT_CALIB_PATH = Path("~/.cache/huggingface/lerobot/calibration/follower.json").expanduser()

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
        # Some firmwares expect ints; we keep them as floats here.
        return float(j["range_min"]), float(j["range_max"])
    except Exception:
        return None

class SO101GripperOnly:
    def __init__(self, port: str, enable: bool, gripper_name: str, use_degrees: bool = True):
        self.enable = enable and (SO101Follower is not None)
        self.port = port
        self.use_degrees = use_degrees
        self.gripper_name = gripper_name
        self.robot: Optional[SO101Follower] = None

    def connect(self):
        if not self.enable:
            return
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            id="follower",
            port=self.port,
            calibration_dir=Path("~/.cache/huggingface/lerobot/calibration").expanduser(),
            use_degrees=self.use_degrees,
            disable_torque_on_disconnect=True,
            max_relative_target=30.0,  # allow larger steps
            cameras={},
            polling_timeout_ms=5,
            connect_timeout_s=3,
        )
        self.robot = SO101Follower(config=cfg)
        self.robot.connect()
        try:
            # Slightly higher accel helps overcome stiction on small moves
            self.robot.bus.configure_motors(maximum_acceleration=120, acceleration=80)
            self.robot.bus.enable_torque()
        except Exception:
            pass
        print(f"[so101] connected on {self.port} (degrees={self.use_degrees})")

    def write_goal(self, value: float, normalize: bool):
        if not (self.robot and self.enable):
            return
        try:
            self.robot.bus.write(WRITE_FIELD, self.gripper_name, float(value), normalize=normalize)
        except Exception as e:
            print(f"[so101] write failed: {e}")

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

def main():
    ap = argparse.ArgumentParser(description="Absolute hand→gripper control (deg-for-deg), gripper-only.")
    # Video / model (GPU by default: leave --device unset so CUDA is used if available)
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--model", type=str, default="wilor")
    ap.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    ap.add_argument("--device", type=str, default=None, help="Leave None to use CUDA if available, or set 'cpu'/'cuda:0'")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)

    # Command range (deg) and optional bias
    ap.add_argument("--cmd-min", type=float, default=0.0)
    ap.add_argument("--cmd-max", type=float, default=90.0)
    ap.add_argument("--offset", type=float, default=0.0)

    # Robot bus
    ap.add_argument("--so101-enable", action="store_true")
    ap.add_argument("--so101-port", type=str, default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00")
    ap.add_argument("--gripper-name", type=str, default=DEFAULT_GRIPPER_NAME)

    # Normalized vs raw write
    ap.add_argument("--raw", action="store_true",
                    help="Write RAW counts (normalize=False) using calibration range_min/max instead of degrees.")
    ap.add_argument("--flip-raw", action="store_true",
                    help="Flip raw mapping (swap range_min/max). Use if raw direction is reversed.")
    ap.add_argument("--calib-path", type=str, default=str(DEFAULT_CALIB_PATH))

    # Debug
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # Pose computer (GPU if available unless you force --device)
    gpc = GripperPoseComputer(device=args.device, model=args.model, hand=args.hand)

    # Robot
    so = SO101GripperOnly(args.so101_port, args.so101_enable, gripper_name=args.gripper_name, use_degrees=True)
    try:
        so.connect()
    except Exception as e:
        print(f"[so101] connect failed: {e}")

    # Load calibration for RAW mode
    calib_range = load_calib_range(Path(args.calib_path), args.gripper_name)
    if args.raw and calib_range is None:
        print(f"[warn] No calibration entry for '{args.gripper_name}' in {args.calib_path}. RAW mode may fail.")
        calib_range = (2048.0, 3072.0)  # safe-ish default span

    if calib_range:
        rmin, rmax = calib_range
        if args.flip-raw:
            rmin, rmax = rmax, rmin
        print(f"[calib] RAW mapping range for '{args.gripper_name}': {rmin} .. {rmax}")

    focal_ratio = 0.9
    cam_t = np.zeros(3, dtype=np.float32)
    target_dt = 1.0 / max(1, args.fps)

    print("\nRealtime absolute control (hand → gripper):")
    if args.raw:
        print("  RAW mode: send counts = lerp( clamp(|hand_deg|+offset, cmd_min, cmd_max), [cmd_min,cmd_max] → [range_min,range_max] )")
    else:
        print("  NORMALIZED mode: send degrees = clamp(|hand_deg|+offset, cmd_min, cmd_max) with normalize=True")
    print("Press Q / ESC to quit.\n")

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            focal_length = focal_ratio * float(frame.shape[1])

            pose_rel: Optional[GripperPose] = gpc.compute_relative_pose(frame, focal_length, cam_t)
            if pose_rel is None:
                cv2.putText(frame, "No hand detected...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("gripper-abs", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
                    break
                # pacing
                dt = time.perf_counter() - t0
                remain = target_dt - dt
                if remain > 0:
                    time.sleep(remain)
                continue

            hand_deg_raw = float(pose_rel.open_degree)
            hand_deg_used = abs(hand_deg_raw) + float(args.offset)
            hand_deg_used = clamp(hand_deg_used, args.cmd_min, args.cmd_max)

            if args.raw and calib_range is not None:
                # Map degrees → raw counts using calibration range
                rmin, rmax = calib_range
                goal_value = lerp(hand_deg_used, (args.cmd_min, args.cmd_max), (rmin, rmax))
                normalize = False
            else:
                # Send degrees directly
                goal_value = hand_deg_used
                normalize = True

            # Write ONLY the gripper joint
            if so.enable and so.robot is not None:
                if args.verbose:
                    unit = "counts" if not normalize else "deg"
                    print(f"[bus] {WRITE_FIELD} {args.gripper_name} = {goal_value:.2f} ({unit})  | hand_raw={hand_deg_raw:.2f}° used={hand_deg_used:.2f}°")
                so.write_goal(goal_value, normalize=normalize)

            # On-screen overlay
            y = 28
            cv2.putText(frame, f"hand_open_degree(raw): {hand_deg_raw:7.2f}°", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2); y += 28
            mode_txt = "RAW counts" if args.raw else "deg (normalized)"
            cv2.putText(frame, f"cmd ({mode_txt}): {goal_value:7.2f}", (20, y),
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
