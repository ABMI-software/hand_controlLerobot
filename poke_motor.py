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
    "gripper":        (-180,180),   # hand open degree (we clamp separately)
}

SAFE_RANGE = {
    "x": (0.13, 0.36),
    "y": (-0.23, 0.23),
    "z": (0.008, 0.25),
    "g": (0, 110),
}

# ---------------- Gripper & (optional) arm conditioning ----------------
# These are small and safe by defgainault — TUNED SLOWER
GRIPPER_DEADBAND_DEG = 1.0       # ignore tiny noise around 0°
GRIPPER_ALPHA = 1.4              # EMA smoothing for open_degree (0..1); higher = snappier
GRIPPER_MAX_STEP_DEG = 4.0       # cap per-frame change (smaller = slower)
GRIPPER_MIN_SEND_DELTA = 1.0     # only update last-sent tracker if changed at least this much

# Arm smoothing: ENABLED and gentle
ARM_ALPHA = 0.2035                 # 0.0 → off; 0.1..0.3 → light smoothing
ARM_MAX_STEP_DEG = 1.0           # per-frame slew limit for each arm joint (smaller = slower)

# Runtime state (initialized in main)
_grip_ema = None
_grip_last_sent = None
_arm_ema = None
_arm_last_sent = None

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

# ------------------------ XYZ → 5 joints mapping (no IK) ----------------------
def xyz_to_joints_deg(
    xyz: np.ndarray,
    *,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = True,
    x_gain: float = 0.2,
    y_gain: float = 0.2,
    z_gain: float = 0.2,
) -> np.ndarray:
    """
    Map XYZ within SAFE_RANGE → 5 arm joints (deg) in your robot’s naming:
      y → shoulder_pan (yaw)
      x → shoulder_lift (pitch)
      z → elbow_flex (pitch)
      (wrist_roll held near 0)
      z → wrist_flex (pitch)
    No IK; with per-axis invert & gain.
    """
    x, y, z = xyz.tolist()

    # center-around-midpoint then apply gain
    def _gain(v, rng, g):
        mid = 0.5 * (rng[0] + rng[1])
        return mid + (v - mid) * max(0.0, g)

    x = _gain(x, SAFE_RANGE["x"], x_gain)
    y = _gain(y, SAFE_RANGE["y"], y_gain)
    z = _gain(z, SAFE_RANGE["z"], z_gain)

    # destination intervals (optionally flipped)
    dst_pan   = maybe_reverse(JOINT_LIMITS_DEG["shoulder_pan"],  invert_y)  # y
    dst_lift  = maybe_reverse(JOINT_LIMITS_DEG["shoulder_lift"], invert_x)  # x
    dst_elbow = maybe_reverse(JOINT_LIMITS_DEG["elbow_flex"],    invert_z)  # z
    dst_roll  = JOINT_LIMITS_DEG["wrist_roll"]                               # 0-ish
    dst_wflex = maybe_reverse(JOINT_LIMITS_DEG["wrist_flex"],    invert_z)  # z

    j_pan   = map_range(y, SAFE_RANGE["y"], dst_pan)
    j_lift  = map_range(x, SAFE_RANGE["x"], dst_lift)
    j_elbow = map_range(z, SAFE_RANGE["z"], dst_elbow)
    j_roll  = 0.0  # neutral
    j_wflex = map_range(z, SAFE_RANGE["z"], dst_wflex)

    out = np.array([j_pan, j_lift, j_elbow, j_roll, j_wflex], dtype=float)
    # clamp everything
    for i, name in enumerate(JOINT_NAMES[:5]):
        lo, hi = JOINT_LIMITS_DEG[name]
        out[i] = clamp(out[i], lo, hi)
    return out

# ------------------------ SO101 client (per-joint write) ----------------------
class SO101Client:
    """Send joint targets to SO-101 by writing each joint's Goal_Position.
       Arm joints in DEGREES (normalize=True).
       Gripper can be DEGREES (normalize=True) or RAW COUNTS (normalize=False).
    """
    def __init__(
        self,
        port: str,
        enable: bool,
        *,
        use_degrees: bool = True,
        gripper_raw: bool = False,
        raw_min: int = 2000,
        raw_max: int = 2600,
        flip_raw: bool = False,
        cmd_min: float = 0.0,
        cmd_max: float = 90.0,
        offset: float = 0.0,
        raw_delta: int = 2,
        verbose: bool = False,
    ):
        self.port = port
        self.enable = enable and (SO101Follower is not None)
        self.use_degrees = use_degrees

        self.gripper_raw = gripper_raw
        self.raw_min = int(raw_min)
        self.raw_max = int(raw_max)
        self.flip_raw = flip_raw
        self.cmd_min = float(cmd_min)
        self.cmd_max = float(cmd_max)
        self.offset = float(offset)
        self.raw_delta = int(raw_delta)
        self.verbose = verbose

        self.robot = None
        self.last_q6 = None
        self._last_gripper_counts: Optional[int] = None  # to reduce spam

    def _deg_to_counts_gripper(self, deg: float) -> int:
        """Map abs(deg)+offset clamped to [cmd_min..cmd_max] into RAW counts."""
        v = clamp(abs(deg) + self.offset, self.cmd_min, self.cmd_max)
        lo, hi = (self.raw_max, self.raw_min) if self.flip_raw else (self.raw_min, self.raw_max)
        c = map_range(v, (self.cmd_min, self.cmd_max), (lo, hi))
        return int(round(c))

    def connect(self):
        if not self.enable:
            return
        from types import SimpleNamespace

        # Slower: smaller per-command step and lower accel
        calib_dir = None if self.gripper_raw else Path("~/.cache/huggingface/lerobot/calibration").expanduser()
        max_rel = 20.0 if self.gripper_raw else 8.0  # smaller = gentler command deltas

        cfg = SimpleNamespace(
            id="follower",
            port=self.port,
            calibration_dir=calib_dir,
            use_degrees=self.use_degrees,
            disable_torque_on_disconnect=True,
            max_relative_target=max_rel,
            cameras={},
            polling_timeout_ms=5,
            connect_timeout_s=3,
        )
        self.robot = SO101Follower(config=cfg)
        self.robot.connect()
        try:
            # Slower hardware acceleration (was 120/80)
            self.robot.bus.configure_motors(maximum_acceleration=40, acceleration=20)
            self.robot.bus.enable_torque()
        except Exception:
            pass
        print(f"[so101] connected on {self.port} (degrees={self.use_degrees}, raw_gripper={self.gripper_raw})")

    def write_joint_deg(self, name: str, deg: float):
        if not self.robot:
            return
        try:
            self.robot.bus.write(WRITE_FIELD, name, float(deg), normalize=True)
        except Exception as e:
            print(f"[so101] write {name} failed: {e}")

    def write_gripper(self, deg: float):
        if not self.robot:
            return
        try:
            if self.gripper_raw:
                counts = self._deg_to_counts_gripper(deg)
                if self._last_gripper_counts is None or abs(counts - self._last_gripper_counts) >= self.raw_delta:
                    if self.verbose:
                        print(f"[bus] {WRITE_FIELD} gripper = {counts} (counts) | hand_deg={deg:.2f}")
                    self.robot.bus.write(WRITE_FIELD, "gripper", int(counts), normalize=False)  # RAW must be int
                    self._last_gripper_counts = counts
            else:
                if self.verbose:
                    print(f"[bus] {WRITE_FIELD} gripper = {deg:.2f} (deg)")
                self.robot.bus.write(WRITE_FIELD, "gripper", float(deg), normalize=True)
        except Exception as e:
            print(f"[so101] write gripper failed: {e}")

    def set_targets(self, q6: np.ndarray | List[float]):
        if not self.robot:
            return
        # clamp to limits for arm joints
        q6 = list(map(float, q6[:6]))
        q6[0] = clamp(q6[0], *JOINT_LIMITS_DEG["shoulder_pan"])
        q6[1] = clamp(q6[1], *JOINT_LIMITS_DEG["shoulder_lift"])
        q6[2] = clamp(q6[2], *JOINT_LIMITS_DEG["elbow_flex"])
        q6[3] = clamp(q6[3], *JOINT_LIMITS_DEG["wrist_roll"])
        q6[4] = clamp(q6[4], *JOINT_LIMITS_DEG["wrist_flex"])
        q6[0] = -q6[0]  # invert shoulder_pan to match robot convention 
        self.last_q6 = q6

        # Send arm joints (0..4) as DEGREES (normalize=True)
        for name, val in zip(JOINT_NAMES[:5], q6[:5]):
            self.write_joint_deg(name, val)

        # Send GRIPPER (deg or RAW)
        self.write_gripper(q6[5])

    def nudge(self, dwell: float = 0.15, amt: float = 20.0):
        if not (self.robot):
            return
        # small open/close to break stiction
        if self.gripper_raw:
            mid = 0.5 * (self.raw_min + self.raw_max)
            try:
                self.robot.bus.write(WRITE_FIELD, "gripper", int(round(mid - amt)), normalize=False)
                time.sleep(dwell)
                self.robot.bus.write(WRITE_FIELD, "gripper", int(round(mid + amt)), normalize=False)
                time.sleep(dwell)
            except Exception as e:
                print(f"[so101] nudge failed: {e}")
        else:
            try:
                self.robot.bus.write(WRITE_FIELD, "gripper", 0.0, normalize=True)
                time.sleep(dwell)
                self.robot.bus.write(WRITE_FIELD, "gripper", float(min(amt, 90.0)), normalize=True)
                time.sleep(dwell)
            except Exception as e:
                print(f"[so101] nudge failed: {e}")

    def safe_stop(self, steps=20, dt=1/50):
        if not self.robot:
            return
        try:
            cur = np.array(self.last_q6 if self.last_q6 is not None else [0,0,0,0,0,0], float)
            tgt = np.zeros(6, dtype=float)  # go home 0s (arm); gripper will be mapped accordingly
            for _ in range(steps):
                cur = 0.8*cur + 0.2*tgt
                self.set_targets(cur)
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

# Patch BEFORE tracker is created (so its thread uses the proxy)
cv2.imshow = _imshow_proxy
cv2.waitKey = _waitKey_proxy

# -------------- Astra / OpenNI shared-memory camera adapter -------------------
# Uses the shared-memory files produced by `./oni_grabber` in your samples/bin.
# Trigger it by passing --cam-idx -1 (keeps all other behavior intact).
class OpenNISharedMemCapture:
    INFO = "/dev/shm/oni_info.txt"
    COLOR = "/dev/shm/oni_color.rgb"
    TICK  = "/dev/shm/oni_tick.txt"

    def __init__(self):
        self._opened = False
        self._w = None
        self._h = None
        # wait for oni_info.txt to appear
        t0 = time.time()
        while not Path(self.INFO).exists():
            time.sleep(0.02)
            if time.time() - t0 > 5.0:
                break
        if not Path(self.INFO).exists():
            print("[openni-cam] oni_info.txt not found (is oni_grabber running?)")
            return
        txt = Path(self.INFO).read_text()
        # Read color width/height (CW/CH). Depth not needed for RGB stream.
        import re as _re
        cw = int(_re.search(r"CW=(\d+)", txt).group(1))
        ch = int(_re.search(r"CH=(\d+)", txt).group(1))
        self._w, self._h = cw, ch
        self._last_tick = ""
        self._opened = True
        print(f"[openni-cam] Opened shared RGB stream {self._w}x{self._h}")

    def isOpened(self):
        return bool(self._opened)

    def read(self):
        if not self._opened:
            return False, None
        # wait for new tick (new frame)
        tries = 0
        while True:
            try:
                t = Path(self.TICK).read_text().strip()
            except Exception:
                t = self._last_tick
            if t != self._last_tick and t != "":
                self._last_tick = t
                break
            time.sleep(0.002)
            tries += 1
            if tries > 2000:  # ~4s max wait
                break
        # read color buffer
        try:
            buf = Path(self.COLOR).read_bytes()
        except Exception:
            return False, None
        expected = self._h * self._w * 3
        if len(buf) != expected:
            return False, None
        rgb = np.frombuffer(buf, np.uint8).reshape(self._h, self._w, 3)
        bgr = rgb[:, :, ::-1].copy()
        return True, bgr

    def release(self):
        self._opened = False

    # minimal CAP_PROP support used by some pipelines
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w or 0)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h or 0)
        return 0.0

# We’ll monkeypatch cv2.VideoCapture so HandTracker(cam_idx=-1) uses the OpenNI stream.
_ORIG_VIDCAP = cv2.VideoCapture
def _VideoCapture_mux(idx_or_path):
    try:
        use_openni = (isinstance(idx_or_path, int) and idx_or_path == -1) or str(idx_or_path).lower() == "openni"
        if use_openni:
            cap = OpenNISharedMemCapture()
            if cap.isOpened():
                return cap
            # fallback: original if openni not available
    except Exception:
        pass
    return _ORIG_VIDCAP(idx_or_path)
cv2.VideoCapture = _VideoCapture_mux
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def main(
    *,
    quiet: bool = False,
    fps: int = 60,
    model: ModelName = "wilor",
    cam_idx: int = 0,
    hand: str = "right",
    use_scroll: bool = True,         # allow mouse wheel for gripper
    # XYZ→joint mapping options (like your example)
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = True,
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    z_gain: float = 1.0,
    # SO-101
    so101_enable: bool = False,
    so101_port: str | None = None,
    # gripper control options
    gripper_raw: bool = False,
    raw_min: int = 2000,
    raw_max: int = 2600,
    flip_raw: bool = False,
    cmd_min: float = 0.0,
    cmd_max: float = 90.0,
    offset: float = 0.0,
    raw_delta: int = 2,
    verbose: bool = False,
    nudge: bool = False,
    nudge_amt: float = 20.0,
):
    # GUI backend
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            pass

    # init runtime state
    global _grip_ema, _grip_last_sent, _arm_ema, _arm_last_sent
    _grip_ema = None
    _grip_last_sent = None
    _arm_ema = None
    _arm_last_sent = None

    # --- hand tracker (no IK path; we’ll use XYZ mapping) ---
    follower_pos = np.array([0.2, 0.0, 0.1])
    follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix()
    follower_pose = GripperPose(follower_pos, follower_rot, open_degree=5)

    tracker = HandTracker(
        cam_idx=cam_idx,          # pass -1 to use Orbbec/OpenNI via our mux above
        hand=hand,
        model=model,
        urdf_path=None,          # no IK: operate in XYZ
        safe_range=SAFE_RANGE,
        use_scroll=use_scroll,   # scroll wheel controls gripper open_degree
        kf_dt=1 / max(1, fps),
        show_viz=True,           # overlays drawn, but windowing is proxied
    )

    # --- optional SO-101 link ---
    so101 = SO101Client(
        port=so101_port or "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00",
        enable=so101_enable,
        use_degrees=True,
        gripper_raw=gripper_raw,
        raw_min=raw_min,
        raw_max=raw_max,
        flip_raw=flip_raw,
        cmd_min=cmd_min,
        cmd_max=cmd_max,
        offset=offset,
        raw_delta=raw_delta,
        verbose=verbose,
    )
    try:
        so101.connect()
    except Exception as e:
        print(f"[so101] connect failed: {e}")
        so101 = SO101Client("", enable=False)

    # Optional nudge
    if nudge and so101.enable and so101.robot is not None:
        so101.nudge(amt=nudge_amt)

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

            # MAIN thread draws the newest frame
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

            # XYZ → joints (5)
            xyz = get_xyz_from_pose(pose)
            q5 = xyz_to_joints_deg(
                xyz,
                invert_x=invert_x, invert_y=invert_y, invert_z=invert_z,
                x_gain=x_gain, y_gain=y_gain, z_gain=z_gain,
            )

            # (Optional) gentle smoothing for the arm (ENABLED)
            if ARM_ALPHA > 0.0:
                if _arm_ema is None:
                    _arm_ema = q5.copy()
                else:
                    _arm_ema = (1.0 - ARM_ALPHA) * _arm_ema + ARM_ALPHA * q5
                q5 = _arm_ema

            # Arm per-frame slew limiter (keeps steps small = slower/safer)
            if _arm_last_sent is None:
                arm_final = q5.copy()
            else:
                step = q5 - _arm_last_sent
                step = np.clip(step, -ARM_MAX_STEP_DEG, ARM_MAX_STEP_DEG)
                arm_final = _arm_last_sent + step
            _arm_last_sent = arm_final
            q5 = arm_final

            # ---------------- Gripper conditioning ----------------
            # ORIGINAL source: tracker open_degree (includes mouse wheel if --use-scroll)
            g_raw = float(getattr(pose, "open_degree", 0.0))

            # 1) deadband
            g_used = 0.0 if abs(g_raw) < GRIPPER_DEADBAND_DEG else g_raw

            # 2) EMA smoothing
            if _grip_ema is None:
                _grip_ema = g_used
            else:
                _grip_ema = (1.0 - GRIPPER_ALPHA) * _grip_ema + GRIPPER_ALPHA * g_used

            # 3) rate-limit step
            if _grip_last_sent is None:
                g_final = _grip_ema
            else:
                step = _grip_ema - _grip_last_sent
                if step > GRIPPER_MAX_STEP_DEG:
                    step = GRIPPER_MAX_STEP_DEG
                elif step < -GRIPPER_MAX_STEP_DEG:
                    step = -GRIPPER_MAX_STEP_DEG
                g_final = _grip_last_sent + step

            # Build q6
            q6 = np.concatenate([q5, [g_final]])

            # Keep wrist_roll strictly neutral to avoid jitter
            q6[3] = 0.0

            # send to robot
            if so101.enable:
                so101.set_targets(q6)

            # update last-sent gripper tracker if changed enough
            if _grip_last_sent is None or abs(g_final - _grip_last_sent) >= GRIPPER_MIN_SEND_DELTA:
                _grip_last_sent = g_final

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
    ap.add_argument("--fps", type=int, default=30)  # try 20 for gentler updates
    ap.add_argument("--model", type=str, default="wilor")
    ap.add_argument("--cam-idx", type=int, default=0,
                    help="Camera index (OpenCV). Use -1 to read Orbbec/OpenNI from /dev/shm/oni_*.")  # <— note
    ap.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    ap.add_argument("--use-scroll", action="store_true", help="Scroll wheel controls gripper open degree")

    # XYZ mapping options (same spirit as your reference)
    ap.add_argument("--invert-x", action="store_true", help="Invert X→shoulder_lift")
    ap.add_argument("--invert-y", action="store_true", help="Invert Y→shoulder_pan")
    ap.add_argument("--invert-z", action="store_true", help="Invert Z→elbow/wrist_flex (default True)")
    ap.add_argument("--x-gain", type=float, default=0.421620050)
    ap.add_argument("--y-gain", type=float, default=0.421620050)
    ap.add_argument("--z-gain", type=float, default=0.421620050)

    # robot link
    ap.add_argument("--so101-enable", action="store_true")
    ap.add_argument("--so101-port", type=str,
                    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00")

    # --- gripper CLI options ---
    ap.add_argument("--raw", action="store_true", help="Send gripper in RAW counts (normalize=False)")
    ap.add_argument("--raw-min", type=int, default=2000, help="RAW min counts (e.g., closed)")
    ap.add_argument("--raw-max", type=int, default=2600, help="RAW max counts (e.g., open)")
    ap.add_argument("--flip-raw", action="store_true", help="Swap RAW min/max direction")
    ap.add_argument("--cmd-min", type=float, default=0.0, help="Min deg after |deg|+offset")
    ap.add_argument("--cmd-max", type=float, default=90.0, help="Max deg after |deg|+offset")
    ap.add_argument("--offset", type=float, default=0.0, help="Added after abs(deg), before clamp")
    ap.add_argument("--raw-delta", type=int, default=2, help="Only send RAW when counts change by at least this many")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--nudge", action="store_true", help="Nudge gripper at start to break stiction")
    ap.add_argument("--nudge-amt", type=float, default=20.0, help="Nudge amplitude (counts if RAW, deg if normalized)")

    # defaults
    ap.set_defaults(invert_z=True)

    args = ap.parse_args()

    main(
        quiet=args.quiet,
        fps=args.fps,
        model=args.model,
        cam_idx=args.cam_idx,
        hand=args.hand,
        use_scroll=args.use_scroll,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_z=args.invert_z,
        x_gain=args.x_gain,
        y_gain=args.y_gain,
        z_gain=args.z_gain,
        so101_enable=args.so101_enable,
        so101_port=args.so101_port,
        gripper_raw=args.raw,
        raw_min=args.raw_min,
        raw_max=args.raw_max,
        flip_raw=args.flip_raw,
        cmd_min=args.cmd_min,
        cmd_max=args.cmd_max,
        offset=args.offset,
        raw_delta=args.raw_delta,
        verbose=args.verbose,
        nudge=args.nudge,
        nudge_amt=args.nudge_amt,
    )
