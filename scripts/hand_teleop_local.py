from __future__ import annotations

import argparse
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Tuple, List, Optional

import cv2
import matplotlib
import numpy as np

# Optional SciPy: we’ll fall back to a local Euler->matrix if it’s missing
try:
    from scipy.spatial.transform import Rotation as R  # noqa: N817
    _HAVE_SCIPY = True
except Exception:
    R = None
    _HAVE_SCIPY = False

def _euler_zyx_to_matrix(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    """Return 3x3 rotation matrix for ZYX Euler angles (degrees)."""
    z = np.deg2rad(z_deg); y = np.deg2rad(y_deg); x = np.deg2rad(x_deg)
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0,0,1]], float)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], float)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], float)
    return Rz @ Ry @ Rx

# project imports (your package)
from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.hand_pose.factory import ModelName
from hand_teleop.tracking.tracker import HandTracker

# --------------------------- LeRobot (SO-101) ---------------------------------
try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
except Exception:
    SO101Follower = None

# ------------- ROS 2 is imported lazily ONLY when --armr-enable is used -------
def _make_armr5_node(joint_traj_topic: str,
                     degrees_topic: str,
                     gripper_topic: str,
                     open_positions: Optional[List[float]],
                     closed_positions: Optional[List[float]]):
    """Create (rclpy, ArmR5Publisher) or (None, None) if ROS isn't usable here."""
    try:
        import rclpy
        from rclpy.node import Node
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        from std_msgs.msg import Float64MultiArray
        from math import pi

        class ArmR5Publisher(Node):
            JOINT_NAMES_ROS = ['R0_Yaw', 'R1_Pitch', 'R2_Pitch', 'R3_Yaw', 'R4_Pitch']
            def __init__(self):
                super().__init__('armr5_publisher')
                self.traj_pub = self.create_publisher(JointTrajectory, joint_traj_topic, 10)
                self.deg_pub  = self.create_publisher(Float64MultiArray, degrees_topic, 10)
                self.grip_pub = self.create_publisher(Float64MultiArray, gripper_topic, 10)
                self.open_positions = open_positions if open_positions is not None else [0,0,0,0,0,0]
                self.closed_positions = closed_positions if closed_positions is not None else [0.99,-0.99,-0.99,-0.99,0.99,-0.99]

            def publish_arm(self, q5_deg: List[float]):
                # 1) Normal ROS trajectory in radians (what controllers expect)
                traj = JointTrajectory()
                traj.joint_names = self.JOINT_NAMES_ROS
                pt = JointTrajectoryPoint()
                pt.positions = [float(v*np.pi/180.0) for v in q5_deg[:5]]
                pt.time_from_start = Duration(sec=0, nanosec=int(1e9/30))
                traj.points = [pt]
                self.traj_pub.publish(traj)

                # 2) Convenience topic in DEGREES (Float64MultiArray) same order
                msg = Float64MultiArray()
                msg.data = [float(v) for v in q5_deg[:5]]
                self.deg_pub.publish(msg)

            def publish_gripper_open_close(self, open_degree: float, threshold_deg: float = 30.0):
                msg = Float64MultiArray()
                msg.data = self.open_positions if float(open_degree) >= float(threshold_deg) else self.closed_positions
                self.grip_pub.publish(msg)

        return rclpy, ArmR5Publisher
    except Exception as e:
        print(f"[armr] ROS2 disabled (rclpy not available here): {e}")
        return None, None

# --------------------------- joints / limits ----------------------------------
JOINT_NAMES = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_roll","wrist_flex","gripper"]
WRITE_FIELD = "Goal_Position"  # LeRobot bus field we write
READ_FIELD  = "Present_Position"  # best-guess field for reading (may differ on your setup)

JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "shoulder_pan":   (-180,180),
    "shoulder_lift":  (-180,180),
    "elbow_flex":     (-180,180),
    "wrist_roll":     (-180,180),
    "wrist_flex":     (-180,180),
    "gripper":        (-180,180),
}

SAFE_RANGE = {
    "x": (0.13, 0.36),
    "y": (-0.23, 0.23),
    "z": (0.008, 0.25),
    "g": (0, 110),
}

# -------------- Gripper & arm conditioning -----------------------------------
GRIPPER_DEADBAND_DEG = 1.0
GRIPPER_ALPHA = 1.4
GRIPPER_MAX_STEP_DEG = 4.0
GRIPPER_MIN_SEND_DELTA = 1.0

ARM_ALPHA = 0.2035
ARM_MAX_STEP_DEG = 1.0

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
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    z_gain: float = 1.0,
) -> np.ndarray:
    """
    Map XYZ within SAFE_RANGE → 5 arm joints (deg) in your robot’s naming:
      y → shoulder_pan (yaw)
      x → shoulder_lift (pitch)
      z → elbow_flex (pitch)
      (wrist_roll held near 0)
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

    out = np.array([j_pan, j_lift, j_elbow, j_roll, j_wflex], float)
    for i, name in enumerate(JOINT_NAMES[:5]):
        lo, hi = JOINT_LIMITS_DEG[name]
        out[i] = clamp(out[i], lo, hi)
    return out

# ------------------------ SO101 client (per-joint write/read) -----------------
class SO101Client:
    """Send AND (optionally) read joint angles with LeRobot SO-101."""
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
        self._last_gripper_counts: Optional[int] = None

    def _deg_to_counts_gripper(self, deg: float) -> int:
        v = clamp(abs(deg) + self.offset, self.cmd_min, self.cmd_max)
        lo, hi = (self.raw_max, self.raw_min) if self.flip_raw else (self.raw_min, self.raw_max)
        c = map_range(v, (self.cmd_min, self.cmd_max), (lo, hi))
        return int(round(c))

    def connect(self):
        if not self.enable:
            return
        from types import SimpleNamespace
        calib_dir = None if self.gripper_raw else Path("~/.cache/huggingface/lerobot/calibration").expanduser()
        max_rel = 20.0 if self.gripper_raw else 8.0
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
                    self.robot.bus.write(WRITE_FIELD, "gripper", int(counts), normalize=False)
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
        q6 = list(map(float, q6[:6]))
        q6[0] = clamp(q6[0], *JOINT_LIMITS_DEG["shoulder_pan"])
        q6[1] = clamp(q6[1], *JOINT_LIMITS_DEG["shoulder_lift"])
        q6[2] = clamp(q6[2], *JOINT_LIMITS_DEG["elbow_flex"])
        q6[3] = clamp(q6[3], *JOINT_LIMITS_DEG["wrist_roll"])
        q6[4] = clamp(q6[4], *JOINT_LIMITS_DEG["wrist_flex"])
        q6[0] = -q6[0]  # invert pan to match robot convention
        self.last_q6 = q6
        for name, val in zip(JOINT_NAMES[:5], q6[:5]):
            self.write_joint_deg(name, val)
        self.write_gripper(q6[5])

    # ---- Optional feedback (best-effort; works only if bus exposes READ_FIELD) ----
    def read_joint_deg(self, name: str) -> Optional[float]:
        if not self.robot:
            return None
        try:
            return float(self.robot.bus.read(READ_FIELD, name, normalize=True))
        except Exception:
            return None

    def read_all_joints_deg(self) -> Dict[str, Optional[float]]:
        return {name: self.read_joint_deg(name) for name in JOINT_NAMES[:5]}

    def safe_stop(self, steps=20, dt=1/50):
        if not self.robot:
            return
        try:
            cur = np.array(self.last_q6 if self.last_q6 is not None else [0,0,0,0,0,0], float)
            tgt = np.zeros(6, dtype=float)
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
    return -1

cv2.imshow = _imshow_proxy
cv2.waitKey = _waitKey_proxy

# -------------- Astra / OpenNI shared-memory camera adapter -------------------
class OpenNISharedMemCapture:
    INFO  = "/dev/shm/oni_info.txt"
    COLOR = "/dev/shm/oni_color.rgb"
    TICK  = "/dev/shm/oni_tick.txt"

    def __init__(self):
        self._opened = False
        self._w = None
        self._h = None
        t0 = time.time()
        while not Path(self.INFO).exists():
            time.sleep(0.02)
            if time.time() - t0 > 5.0:
                break
        if not Path(self.INFO).exists():
            print("[openni-cam] oni_info.txt not found (is oni_grabber running?)")
            return
        txt = Path(self.INFO).read_text()
        import re as _re
        cw = int(_re.search(r"CW=(\d+)", txt).group(1))
        ch = int(_re.search(r"CH=(\d+)", txt).group(1))
        self._w, self._h = cw, ch
        self._last_tick = ""
        self._opened = True
        print(f"[openni-cam] Opened shared RGB stream {self._w}x{self._h}")

    def isOpened(self): return bool(self._opened)

    def read(self):
        if not self._opened:
            return False, None
        tries = 0
        while True:
            try:
                t = Path(self.TICK).read_text().strip()
            except Exception:
                t = self._last_tick
            if t != self._last_tick and t != "":
                self._last_tick = t
                break
            time.sleep(0.002); tries += 1
            if tries > 2000: break
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

    def release(self): self._opened = False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._w or 0)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h or 0)
        return 0.0

_ORIG_VIDCAP = cv2.VideoCapture
def _VideoCapture_mux(idx_or_path):
    try:
        use_openni = (isinstance(idx_or_path, int) and idx_or_path == -1) or str(idx_or_path).lower() == "openni"
        if use_openni:
            cap = OpenNISharedMemCapture()
            if cap.isOpened():
                return cap
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
    use_scroll: bool = True,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = True,
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    z_gain: float = 1.0,
    # SO-101
    so101_enable: bool = False,
    so101_port: str | None = None,
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
    # ROS2 (optional)
    armr_enable: bool = False,
    armr_joint_traj_topic: str = '/armr5_controller/joint_trajectory',
    armr_degrees_topic: str = '/armr5_controller/joint_degrees',
    armr_gripper_topic: str = '/gripper_controller/commands',
    armr_open_positions: Optional[List[float]] = None,
    armr_closed_positions: Optional[List[float]] = None,
    armr_open_threshold_deg: float = 30.0,
    # debug / UX
    print_joints: bool = False,
):
    # GUI backend
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            pass

    # runtime state
    global _grip_ema, _grip_last_sent, _arm_ema, _arm_last_sent
    _grip_ema = None; _grip_last_sent = None
    _arm_ema = None;  _arm_last_sent = None

    # Hand tracker (no IK)
    follower_pos = np.array([0.2, 0.0, 0.1])
    follower_rot = (R.from_euler("ZYX", [0,45,-90], degrees=True).as_matrix()
                    if _HAVE_SCIPY else _euler_zyx_to_matrix(0,45,-90))
    follower_pose = GripperPose(follower_pos, follower_rot, open_degree=5)

    tracker = HandTracker(
        cam_idx=cam_idx,
        hand=hand,
        model=model,
        urdf_path=None,
        safe_range=SAFE_RANGE,
        use_scroll=use_scroll,
        kf_dt=1 / max(1, fps),
        show_viz=True,
    )

    # SO-101 link
    so101 = SO101Client(
        port=so101_port or "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00",
        enable=so101_enable,
        use_degrees=True,
        gripper_raw=gripper_raw,
        raw_min=raw_min, raw_max=raw_max, flip_raw=flip_raw,
        cmd_min=cmd_min, cmd_max=cmd_max, offset=offset,
        raw_delta=raw_delta, verbose=verbose,
    )
    try:
        so101.connect()
    except Exception as e:
        print(f"[so101] connect failed: {e}")
        so101 = SO101Client("", enable=False)

    # ROS2 (optional)
    armr_node = None; rclpy = None; ArmR5Publisher = None
    if armr_enable:
        rclpy, ArmR5Publisher = _make_armr5_node(
            joint_traj_topic=armr_joint_traj_topic,
            degrees_topic=armr_degrees_topic,
            gripper_topic=armr_gripper_topic,
            open_positions=armr_open_positions,
            closed_positions=armr_closed_positions,
        )
        if rclpy and ArmR5Publisher:
            try:
                rclpy.init()
                armr_node = ArmR5Publisher()
                print("[armr] ROS2 armr5_controller outputs enabled (traj=radians, degrees topic provided).")
            except Exception as e:
                print(f"[armr] ROS2 init failed: {e}")
                armr_node = None

    # Optional nudge
    if nudge and so101.enable and so101.robot is not None:
        so101.nudge(amt=nudge_amt)

    # Real HighGUI window (main thread)
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

            # Draw newest frame in main thread
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

            # XYZ → joints (5) in degrees
            xyz = get_xyz_from_pose(pose)
            q5 = xyz_to_joints_deg(
                xyz,
                invert_x=invert_x, invert_y=invert_y, invert_z=invert_z,
                x_gain=x_gain, y_gain=y_gain, z_gain=z_gain,
            )

            # smoothing
            if ARM_ALPHA > 0.0:
                if _arm_ema is None: _arm_ema = q5.copy()
                else: _arm_ema = (1.0 - ARM_ALPHA) * _arm_ema + ARM_ALPHA * q5
                q5 = _arm_ema

            # slew limit
            if _arm_last_sent is None:
                arm_final = q5.copy()
            else:
                step = q5 - _arm_last_sent
                step = np.clip(step, -ARM_MAX_STEP_DEG, ARM_MAX_STEP_DEG)
                arm_final = _arm_last_sent + step
            _arm_last_sent = arm_final
            q5 = arm_final

            # gripper conditioning
            g_raw = float(getattr(pose, "open_degree", 0.0))
            g_used = 0.0 if abs(g_raw) < GRIPPER_DEADBAND_DEG else g_raw
            if _grip_ema is None: _grip_ema = g_used
            else: _grip_ema = (1.0 - GRIPPER_ALPHA) * _grip_ema + GRIPPER_ALPHA * g_used
            if _grip_last_sent is None: g_final = _grip_ema
            else:
                s = _grip_ema - _grip_last_sent
                s = max(-GRIPPER_MAX_STEP_DEG, min(GRIPPER_MAX_STEP_DEG, s))
                g_final = _grip_last_sent + s

            # Build q6 (deg)
            q6 = np.concatenate([q5, [g_final]])
            q6[3] = 0.0  # wrist_roll neutral

            # ---- PRINT the angles (deg) every frame ----
            if print_joints:
                print(
                    "CMD deg | "
                    f"shoulder_pan:{q5[0]:+7.2f}  shoulder_lift:{q5[1]:+7.2f}  "
                    f"elbow_flex:{q5[2]:+7.2f}  wrist_roll:{q5[3]:+7.2f}  wrist_flex:{q5[4]:+7.2f}  "
                    f"| gripper:{g_final:+6.2f}"
                )

            # send to SO-101
            if so101.enable:
                so101.set_targets(q6)

                # Optional feedback print (best-effort)
                if print_joints:
                    fb = so101.read_all_joints_deg()
                    if any(v is not None for v in fb.values()):
                        line = "FB  deg | " + "  ".join(f"{k}:{(v if v is not None else float('nan')):+7.2f}" for k,v in fb.items())
                        print(line)

            # ROS 2 publishing (traj in radians + degrees on side channel)
            if armr_node is not None and 'rclpy' in locals() and rclpy is not None:
                try:
                    armr_node.publish_arm(list(q5[:5]))  # deg
                    armr_node.publish_gripper_open_close(g_final, threshold_deg=armr_open_threshold_deg)
                    if print_joints:
                        print("[ROS2] published /armr5_controller/joint_trajectory (rad) + /armr5_controller/joint_degrees (deg)")
                    rclpy.spin_once(armr_node, timeout_sec=0.0)
                except Exception as e:
                    print(f"[armr] publish failed: {e}")

            if _grip_last_sent is None or abs(g_final - _grip_last_sent) >= GRIPPER_MIN_SEND_DELTA:
                _grip_last_sent = g_final

            # FPS + pacing
            dt = time.perf_counter() - t0
            inst = (1.0/dt) if dt > 0 else fps
            ema_fps = inst if ema_fps is None else 0.9*ema_fps + 0.1*inst
            if not quiet:
                pairs = ", ".join(f"{n}:{v:.1f}" for n, v in zip(JOINT_NAMES, q6))
                print(f"XYZ {np.round(xyz,3)} | {pairs} | FPS: {ema_fps:.1f}")

            remain = target_dt - (time.perf_counter() - t0)
            if remain > 0:
                time.sleep(remain)

    finally:
        try: tracker.cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try:
            if so101.enable:
                so101.safe_stop(steps=20, dt=1/50)
        except Exception as e:
            print(f"[so101] safe stop error: {e}")
        try:
            if armr_node is not None and 'rclpy' in locals() and rclpy is not None:
                armr_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        except Exception:
            pass

    cv2.imshow = _original_imshow
    cv2.waitKey = _original_waitKey


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", type=str, default="wilor")
    ap.add_argument("--cam-idx", type=int, default=0,
                    help="Camera index (OpenCV). Use -1 to read Orbbec/OpenNI from /dev/shm/oni_*.")
    ap.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    ap.add_argument("--use-scroll", action="store_true", help="Scroll wheel controls gripper open degree")

    # XYZ mapping (same spirit as yours)
    ap.add_argument("--invert-x", action="store_true", help="Invert X→shoulder_lift")
    ap.add_argument("--invert-y", action="store_true", help="Invert Y→shoulder_pan")
    ap.add_argument("--invert-z", action="store_true", help="Invert Z→elbow/wrist_flex (default True)")
    ap.add_argument("--x-gain", type=float, default=0.421620050)
    ap.add_argument("--y-gain", type=float, default=0.421620050)
    ap.add_argument("--z-gain", type=float, default=0.421620050)

    # SO-101
    ap.add_argument("--so101-enable", action="store_true")
    ap.add_argument("--so101-port", type=str,
                    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00")
    ap.add_argument("--raw", action="store_true", help="Send gripper in RAW counts (normalize=False)")
    ap.add_argument("--raw-min", type=int, default=2000)
    ap.add_argument("--raw-max", type=int, default=2600)
    ap.add_argument("--flip-raw", action="store_true")
    ap.add_argument("--cmd-min", type=float, default=0.0)
    ap.add_argument("--cmd-max", type=float, default=90.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--raw-delta", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--nudge", action="store_true")
    ap.add_argument("--nudge-amt", type=float, default=20.0)

    # ROS 2 (optional)
    ap.add_argument("--armr-enable", action="store_true",
                    help="Publish to /armr5_controller (traj: radians) and /armr5_controller/joint_degrees (degrees).")
    ap.add_argument("--armr-joint-traj-topic", type=str, default="/armr5_controller/joint_trajectory")
    ap.add_argument("--armr-degrees-topic", type=str, default="/armr5_controller/joint_degrees")
    ap.add_argument("--armr-gripper-topic", type=str, default="/gripper_controller/commands")
    ap.add_argument("--armr-open-threshold-deg", type=float, default=30.0)
    ap.add_argument("--armr-open-positions", type=float, nargs=6, default=None,
                    help="6-float preset for 'open' (Float64MultiArray).")
    ap.add_argument("--armr-closed-positions", type=float, nargs=6, default=None,
                    help="6-float preset for 'closed' (Float64MultiArray).")

    # Debug
    ap.add_argument("--print-joints", action="store_true", help="Print commanded joints (deg) and gripper each frame")

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
        armr_enable=args.armr_enable,
        armr_joint_traj_topic=args.armr_joint_traj_topic,
        armr_degrees_topic=args.armr_degrees_topic,
        armr_gripper_topic=args.armr_gripper_topic,
        armr_open_positions=args.armr_open_positions,
        armr_closed_positions=args.armr_closed_positions,
        armr_open_threshold_deg=args.armr_open_threshold_deg,
        print_joints=args.print_joints,
    )
