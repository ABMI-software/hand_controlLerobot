#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Tuple, List, Optional

import cv2
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.hand_pose.factory import ModelName
from hand_teleop.tracking.tracker import HandTracker

# --------------------------- LeRobot (SO-101) ---------------------------------
try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
except Exception:
    SO101Follower = None

# Your robot joint names (in the order we command them) + gripper last
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex", "gripper"]
WRITE_FIELD = "Goal_Position"  # confirmed working

# Per-joint safe limits (deg) — adjust if needed
JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "shoulder_pan":   (-180,180),   # yaw
    "shoulder_lift":  ( -180,180),  # pitch
    "elbow_flex":     ( -180,180),  # pitch
    "wrist_roll":     ( -180,180),  # yaw
    "wrist_flex":     ( -180,180),  # pitch
    "gripper":        (  -180,180), # open degree (pre-clamped later)
}

SAFE_RANGE = {
    "x": (0.13, 0.36),
    "y": (-0.23, 0.23),
    "z": (0.008, 0.25),
    "g": (2, 90),
}

# --------------------------- helpers ------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))

def map_range(v: float, src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    s0, s1 = src; d0, d1 = dst
    if s1 == s0:
        return d0
    t = (v - s0) / (s1 - s0)
    t = clamp(t, 0.0, 1.0)
    return d0 + t * (d1 - d0)

def maybe_reverse(rng: Tuple[float, float], invert: bool) -> Tuple[float, float]:
    lo, hi = rng
    return (hi, lo) if invert else (lo, hi)

def get_xyz_from_pose(pose: GripperPose) -> np.ndarray:
    return np.asarray(pose.pos if hasattr(pose, "pos") else getattr(pose, "position"), dtype=float)

def load_calib_range(calib_path: Path, joint_name: str) -> Optional[Tuple[float, float]]:
    try:
        import json
        data = json.loads(calib_path.read_text())
        j = data.get(joint_name)
        if not j:
            return None
        return float(j["range_min"]), float(j["range_max"])
    except Exception:
        return None

# ------------------------ XYZ → 5 joints mapping (no IK) ----------------------
def xyz_to_joints_deg(
    xyz: np.ndarray,
    *,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = True,
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    z_gain: float = 1.0,
) -> np.ndarray:
    """
    Map XYZ within SAFE_RANGE → 5 arm joints (deg):
      y → shoulder_pan (yaw)
      x → shoulder_lift (pitch)
      z → elbow_flex (pitch)
      wrist_roll = 0
      z → wrist_flex (pitch)
    """
    x, y, z = xyz.tolist()

    def _gain(v, rng, g):
        mid = 0.5 * (rng[0] + rng[1])
        return mid + (v - mid) * max(0.0, g)

    x = _gain(x, SAFE_RANGE["x"], x_gain)
    y = _gain(y, SAFE_RANGE["y"], y_gain)
    z = _gain(z, SAFE_RANGE["z"], z_gain)

    dst_pan   = maybe_reverse(JOINT_LIMITS_DEG["shoulder_pan"],  invert_y)
    dst_lift  = maybe_reverse(JOINT_LIMITS_DEG["shoulder_lift"], invert_x)
    dst_elbow = maybe_reverse(JOINT_LIMITS_DEG["elbow_flex"],    invert_z)
    dst_roll  = JOINT_LIMITS_DEG["wrist_roll"]
    dst_wflex = maybe_reverse(JOINT_LIMITS_DEG["wrist_flex"],    invert_z)

    j_pan   = map_range(y, SAFE_RANGE["y"], dst_pan)
    j_lift  = map_range(x, SAFE_RANGE["x"], dst_lift)
    j_elbow = map_range(z, SAFE_RANGE["z"], dst_elbow)
    j_roll  = 0.0
    j_wflex = map_range(z, SAFE_RANGE["z"], dst_wflex)

    out = np.array([j_pan, j_lift, j_elbow, j_roll, j_wflex], dtype=float)
    for i, name in enumerate(JOINT_NAMES[:5]):
        lo, hi = JOINT_LIMITS_DEG[name]
        out[i] = clamp(out[i], lo, hi)
    return out

# ------------------------ SO101 client (per-joint write) ----------------------
class SO101Client:
    """Send joint targets to SO-101 by writing each joint's Goal_Position.
       Joints 1..5 are sent in degrees (normalize=True).
       Gripper can be degrees OR RAW counts with mapping/flip/delta/nudge like your gripper RT script.
    """
    def __init__(
        self,
        port: str,
        enable: bool,
        use_degrees: bool = True,
        *,
        # gripper behavior
        gripper_raw: bool = False,
        gripper_cmd_min: float = 0.0,
        gripper_cmd_max: float = 90.0,
        gripper_offset: float = 0.0,
        gripper_calib_path: Optional[Path] = None,
        gripper_flip_raw: bool = False,
        gripper_raw_min: Optional[float] = None,
        gripper_raw_max: Optional[float] = None,
        gripper_raw_delta: int = 2,
        verbose: bool = False,
    ):
        self.port = port
        self.enable = enable and (SO101Follower is not None)
        self.use_degrees = use_degrees
        self.robot = None
        self.last_q6 = None

        # gripper config
        self.gripper_raw = bool(gripper_raw)
        self.g_cmd_min = float(gripper_cmd_min)
        self.g_cmd_max = float(gripper_cmd_max)
        self.g_offset  = float(gripper_offset)
        self.g_calib_path = gripper_calib_path
        self.g_flip_raw = bool(gripper_flip_raw)
        self.g_raw_min = None if gripper_raw_min is None else float(gripper_raw_min)
        self.g_raw_max = None if gripper_raw_max is None else float(gripper_raw_max)
        self.g_raw_delta = int(gripper_raw_delta)
        self.g_last_raw_sent: Optional[int] = None
        self.verbose = bool(verbose)

        self._raw_range: Optional[Tuple[float, float]] = None

    def _resolve_gripper_raw_range(self) -> Tuple[float, float]:
        if self.g_raw_min is not None and self.g_raw_max is not None:
            rmin, rmax = self.g_raw_min, self.g_raw_max
        else:
            r = None
            if self.g_calib_path is not None:
                r = load_calib_range(self.g_calib_path, "gripper")
            if r is None:
                r = (1900.0, 2600.0)
                print(f"[warn] No gripper calibration in {self.g_calib_path} and no --raw-min/--raw-max; using 1900..2600.")
            rmin, rmax = r
        if self.g_flip_raw:
            rmin, rmax = rmax, rmin
        return float(rmin), float(rmax)

    def connect(self, *, do_nudge: bool = False, nudge_amt: float = 20.0):
        if not self.enable:
            return
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            id="follower",
            port=self.port,
            calibration_dir=Path("~/.cache/huggingface/lerobot/calibration").expanduser(),
            use_degrees=self.use_degrees,
            disable_torque_on_disconnect=True,
            max_relative_target=15.0,
            cameras={},
            polling_timeout_ms=5,
            connect_timeout_s=3,
        )
        self.robot = SO101Follower(config=cfg)
        self.robot.connect()
        try:
            self.robot.bus.configure_motors(maximum_acceleration=80, acceleration=60)
            self.robot.bus.enable_torque()
        except Exception:
            pass

        if self.gripper_raw:
            self._raw_range = self._resolve_gripper_raw_range()
            rmin, rmax = self._raw_range
            print(f"[calib] Gripper RAW counts range: {rmin} .. {rmax}")

        print(f"[so101] connected on {self.port} (degrees={self.use_degrees}, gripper_raw={self.gripper_raw})")

        if do_nudge:
            try:
                if self.gripper_raw:
                    rmin, rmax = self._raw_range if self._raw_range else (1900.0, 2600.0)
                    mid = 0.5 * (rmin + rmax)
                    self._write_gripper_raw(int(round(mid - nudge_amt)))
                    time.sleep(0.15)
                    self._write_gripper_raw(int(round(mid + nudge_amt)))
                    time.sleep(0.15)
                else:
                    self._write_gripper_deg(max(0.0, 0.0))
                    time.sleep(0.15)
                    self._write_gripper_deg(float(nudge_amt))
                    time.sleep(0.15)
            except Exception as e:
                print(f"[so101] nudge failed: {e}")

    # gripper write helpers
    def _write_gripper_deg(self, degrees_val: float):
        if not self.robot:
            return
        try:
            self.robot.bus.write(WRITE_FIELD, "gripper", float(degrees_val), normalize=True)
        except Exception as e:
            print(f"[so101] write gripper (deg) failed: {e}")

    def _write_gripper_raw(self, raw_counts: int):
        if not self.robot:
            return
        try:
            self.robot.bus.write(WRITE_FIELD, "gripper", int(raw_counts), normalize=False)
        except Exception as e:
            print(f"[so101] write gripper (RAW) failed: {e}")

    def _process_gripper_command(self, hand_open_deg: float) -> Tuple[str, float]:
        used_deg = clamp(abs(float(hand_open_deg)) + self.g_offset, self.g_cmd_min, self.g_cmd_max)
        if not self.gripper_raw:
            return ("deg", used_deg)
        if self._raw_range is None:
            self._raw_range = self._resolve_gripper_raw_range()
        rmin, rmax = self._raw_range
        raw = int(round(map_range(used_deg, (self.g_cmd_min, self.g_cmd_max), (rmin, rmax))))
        return ("raw", raw)

    def set_targets(self, q6: np.ndarray | List[float]):
        if not self.robot:
            return
        q6 = list(map(float, q6[:6]))
        # clamp 5 arm joints
        q6[0] = clamp(q6[0], *JOINT_LIMITS_DEG["shoulder_pan"])
        q6[1] = clamp(q6[1], *JOINT_LIMITS_DEG["shoulder_lift"])
        q6[2] = clamp(q6[2], *JOINT_LIMITS_DEG["elbow_flex"])
        q6[3] = clamp(q6[3], *JOINT_LIMITS_DEG["wrist_roll"])
        q6[4] = clamp(q6[4], *JOINT_LIMITS_DEG["wrist_flex"])
        self.last_q6 = q6

        # write 5 arm joints (degrees)
        for name, val in zip(JOINT_NAMES[:5], q6[:5]):
            try:
                self.robot.bus.write(WRITE_FIELD, name, float(val), normalize=True)
            except Exception as e:
                print(f"[so101] write {name} failed: {e}")

        # write gripper (deg or RAW with delta)
        mode, g_val = self._process_gripper_command(hand_open_deg=q6[5])
        try:
            if mode == "deg":
                if self.verbose:
                    print(f"[bus] {WRITE_FIELD} gripper = {g_val:.2f} deg")
                self._write_gripper_deg(g_val)
            else:
                gv = int(g_val)
                send = (self.g_last_raw_sent is None) or (abs(gv - self.g_last_raw_sent) >= self.g_raw_delta)
                if send:
                    if self.verbose:
                        print(f"[bus] {WRITE_FIELD} gripper = {gv} counts (Δ≥{self.g_raw_delta})")
                    self._write_gripper_raw(gv)
                    self.g_last_raw_sent = gv
        except Exception as e:
            print(f"[so101] gripper write failed: {e}")

    def safe_stop(self, steps=20, dt=1/50):
        if not self.robot:
            return
        try:
            cur = np.array(self.last_q6 if self.last_q6 is not None else [0,0,0,0,0,0], float)
            tgt = np.zeros(6, dtype=float)  # home (all zeros)
            for _ in range(steps):
                cur = 0.8*cur + 0.2*tgt
                safe_q6 = cur.copy()
                safe_q6[5] = 0.0  # send gripper to 0 deg for safety
                self.set_targets(safe_q6)
                time.sleep(dt)
        finally:
            try:
                self.robot.bus.disable_torque()
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass
            print("[so101] disconnected")

# -------------------- Thread-safe OpenCV window bridge ------------------------
_original_imshow = cv2.imshow
_original_waitKey = cv2.waitKey
_frames: Dict[str, np.ndarray] = {}
_frame_queue: "Queue[tuple[str, np.ndarray]]" = Queue(maxsize=2)

def _imshow_proxy(win: str, frame: np.ndarray):
    try:
        while not _frame_queue.empty():
            _frame_queue.get_nowait()
    except Exception:
        pass
    _frames[win] = frame
    try:
        _frame_queue.put_nowait((win, frame))
    except Exception:
        pass
    return True

def _waitKey_proxy(ms: int):
    return -1  # worker doesn’t handle keys

# Patch BEFORE tracker is created
cv2.imshow = _imshow_proxy
cv2.waitKey = _waitKey_proxy

# -----------------------------------------------------------------------------
def main(
    *,
    quiet: bool = False,
    fps: int = 60,
    model: ModelName = "wilor",
    cam_idx: int = 0,
    hand: str = "right",
    use_scroll: bool = False,        # default OFF: gripper uses hand angle, not mouse
    # XYZ mapping options
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = True,
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    z_gain: float = 1.0,
    # SO-101
    so101_enable: bool = False,
    so101_port: str | None = None,
    # gripper behavior (same semantics as your gripper RT script)
    g_cmd_min: float = 0.0,
    g_cmd_max: float = 90.0,
    g_offset: float = 0.0,
    g_raw: bool = False,
    g_flip_raw: bool = False,
    g_raw_min: Optional[float] = None,
    g_raw_max: Optional[float] = None,
    g_calib_path: Optional[Path] = None,
    g_raw_delta: int = 2,
    g_nudge: bool = False,
    g_nudge_amt: float = 20.0,
    verbose: bool = False,
):
    # GUI backend choice
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            pass

    # --- hand tracker (no IK: operate in XYZ) ---
    follower_pos = np.array([0.2, 0.0, 0.1])
    follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix()
    follower_pose = GripperPose(follower_pos, follower_rot, open_degree=5)

    tracker = HandTracker(
        cam_idx=cam_idx,
        hand=hand,
        model=model,
        urdf_path=None,
        safe_range=SAFE_RANGE,
        use_scroll=use_scroll,   # default False
        kf_dt=1 / max(1, fps),
        show_viz=True,
    )

    # --- SO-101 link ---
    so101 = SO101Client(
        port=so101_port or "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00",
        enable=so101_enable,
        use_degrees=True,
        gripper_raw=g_raw,
        gripper_cmd_min=g_cmd_min,
        gripper_cmd_max=g_cmd_max,
        gripper_offset=g_offset,
        gripper_calib_path=g_calib_path,
        gripper_flip_raw=g_flip_raw,
        gripper_raw_min=g_raw_min,
        gripper_raw_max=g_raw_max,
        gripper_raw_delta=g_raw_delta,
        verbose=verbose,
    )
    try:
        so101.connect(do_nudge=g_nudge, nudge_amt=g_nudge_amt)
    except Exception as e:
        print(f"[so101] connect failed: {e}")
        so101 = SO101Client("", enable=False)

    # Create real HighGUI window in MAIN thread (if available)
    try:
        _original_imshow("hand-teleop", np.zeros((10, 10, 3), dtype=np.uint8))
        _original_waitKey(1)
    except Exception:
        print("[viz] OpenCV GUI not available. Camera window will be skipped.")

    target_dt = 1.0 / fps
    ema_fps = None

    try:
        while tracker.cap.isOpened():
            t0 = time.perf_counter()

            try:
                pose = tracker.read_hand_state(follower_pose)
            except RuntimeError as e:
                if not quiet:
                    print(f"[ERROR] {e}")
                break

            # MAIN thread draws newest frame
            try:
                got = False
                while not _frame_queue.empty():
                    last_name, last_frame = _frame_queue.get_nowait()
                    got = True
                if got:
                    _original_imshow(last_name, last_frame)
                    k = _original_waitKey(1) & 0xFF
                    if k in (27, ord("q")):
                        break
            except Exception:
                pass

            # XYZ → 5 joints
            xyz = get_xyz_from_pose(pose)
            q5 = xyz_to_joints_deg(
                xyz,
                invert_x=invert_x, invert_y=invert_y, invert_z=invert_z,
                x_gain=x_gain, y_gain=y_gain, z_gain=z_gain,
            )

            # gripper source is the hand open degree (no scroll unless --use-scroll)
            g = float(getattr(pose, "open_degree", 0.0))
            q6 = np.concatenate([q5, [g]])

            # --------- HARD-WIRED: reverse ONLY shoulder_pan (q6[0]) ----------
            q6[0] = -q6[0]
            # -------------------------------------------------------------------

            # send to robot
            if so101.enable:
                so101.set_targets(q6)

            # status / pacing
            dt = time.perf_counter() - t0
            inst = (1.0 / dt) if dt > 0 else fps
            ema_fps = inst if ema_fps is None else 0.9 * ema_fps + 0.1 * inst
            if not quiet:
                pairs = ", ".join(f"{n}:{v:.1f}" for n, v in zip(JOINT_NAMES, q6))
                print(f"XYZ {np.round(xyz,3)} | {pairs} | FPS: {ema_fps:.1f}")

            remain = target_dt - (time.perf_counter() - t0)
            if remain > 0:
                time.sleep(remain)

    finally:
        tracker.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if so101.enable:
                so101.safe_stop(steps=20, dt=1/50)
        except Exception as e:
            print(f"[so101] safe stop error: {e}")

    # restore OpenCV (optional)
    cv2.imshow = _original_imshow
    cv2.waitKey = _original_waitKey


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", type=str, default="wilor")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--hand", type=str, default="right", choices=["left", "right"])

    # XYZ mapping options
    ap.add_argument("--invert-x", action="store_true", help="Invert X→shoulder_lift")
    ap.add_argument("--invert-y", action="store_true", help="Invert Y→shoulder_pan")
    ap.add_argument("--invert-z", action="store_true", help="Invert Z→elbow/wrist_flex (default True)")
    ap.add_argument("--x-gain", type=float, default=1.0)
    ap.add_argument("--y-gain", type=float, default=1.0)
    ap.add_argument("--z-gain", type=float, default=1.0)

    # robot link
    ap.add_argument("--so101-enable", action="store_true")
    ap.add_argument("--so101-port", type=str,
                    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00")

    # gripper options (same as your gripper-only CLI)
    ap.add_argument("--cmd-min", type=float, default=0.0, help="Gripper min degrees after abs()")
    ap.add_argument("--cmd-max", type=float, default=90.0, help="Gripper max degrees after abs()")
    ap.add_argument("--offset",  type=float, default=0.0, help="Gripper offset added before clamp")
    ap.add_argument("--raw", action="store_true",
                    help="Gripper RAW counts mode (normalize=False) with degree→count mapping.")
    ap.add_argument("--flip-raw", action="store_true",
                    help="Flip RAW mapping direction (swap min/max).")
    ap.add_argument("--raw-min", type=float, default=None, help="Override RAW range_min (counts)")
    ap.add_argument("--raw-max", type=float, default=None, help="Override RAW range_max (counts)")
    ap.add_argument("--calib-path", type=Path, default=Path("~/.cache/huggingface/lerobot/calibration/follower.json").expanduser(),
                    help="Calibration JSON with per-joint range_min/range_max")
    ap.add_argument("--raw-delta", type=int, default=2, help="Only send RAW when counts change by at least this")
    ap.add_argument("--nudge", action="store_true", help="Send a tiny open/close on connect")
    ap.add_argument("--nudge-amt", type=float, default=20.0, help="Nudge amplitude (deg or counts in RAW)")
    ap.add_argument("--verbose", action="store_true")

    # optional: mouse scroll for gripper (OFF unless requested)
    ap.add_argument("--use-scroll", action="store_true", help="Scroll wheel controls gripper open degree")

    ap.set_defaults(invert_z=True)

    args = ap.parse_args()

    main(
        quiet=args.quiet,
        fps=args.fps,
        model=args.model,
        cam_idx=args.cam_idx,
        hand=args.hand,
        use_scroll=args.use_scroll,  # leave out this flag to behave like gripper-only
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_z=args.invert_z,
        x_gain=args.x_gain,
        y_gain=args.y_gain,
        z_gain=args.z_gain,
        so101_enable=args.so101_enable,
        so101_port=args.so101_port,
        g_cmd_min=args.cmd_min,
        g_cmd_max=args.cmd_max,
        g_offset=args.offset,
        g_raw=args.raw,
        g_flip_raw=args.flip_raw,
        g_raw_min=args.raw_min,
        g_raw_max=args.raw_max,
        g_calib_path=args.calib_path,
        g_raw_delta=args.raw_delta,
        g_nudge=args.nudge,
        g_nudge_amt=args.nudge_amt,
        verbose=args.verbose,
    )
