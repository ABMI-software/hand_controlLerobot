import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.hand_pose.factory import ModelName
from hand_teleop.kinematics.robot_visualisation import RobotVisualisation
from hand_teleop.tracking.tracker import HandTracker


# --------------------------------------------------------------------- #
def make_target_transform(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Return a 4×4 homogeneous transform (world → gripper)."""
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def main(quiet=False, fps=60, test_joint=True, model: ModelName = "wilor"):
    urdf_path = "so100" if test_joint else None
    tracker = HandTracker(cam_idx=1, hand="right", model=model, urdf_path=urdf_path)

    # --- follower pose in SE(3) ------------------------------------------------
    follower_pos = np.array([0.2, 0, 0.1])
    follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix()
    follower_pose = GripperPose(follower_pos, follower_rot, open_degree=5)

    # --- if joint-space test: pose → joints via IK -----------------------------
    if test_joint:
        kin = tracker.robot_kin
        q0 = np.array([0,2,2,0,0])
        follower_joint = kin.ik(q0, make_target_transform(follower_pos, follower_rot))[:5]
        viz = RobotVisualisation(kin, "so100")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(follower_joint)
    # -------------------------------------------------------------------------
    target_dt = 1.0 / fps
    ema_fps, correction = None, 0.0  # for adaptive sleep

    while tracker.cap.isOpened():
        t0 = time.perf_counter()

        try:
            if test_joint:
                q = tracker.read_hand_state_joint(follower_joint)
            else:
                pose = tracker.read_hand_state(follower_pose)
        except RuntimeError as e:
            if not quiet:
                print(f"[ERROR] {e}")
            break

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        # adaptive sleep to hold target FPS
        remain = target_dt - (time.perf_counter() - t0) - correction
        if remain > 0:
            time.sleep(remain)

        dt = time.perf_counter() - t0
        inst_fps = 1.0 / dt
        ema_fps = inst_fps if ema_fps is None else 0.9 * ema_fps + 0.1 * inst_fps
        correction = np.clip(correction - 5e-4 * (ema_fps - fps), 0, 0.02)

        if not quiet:
            if test_joint:
                print(f"Joints: {np.round(q, 3)} | FPS: {ema_fps:.1f}")
                viz.draw(ax, q)
                plt.pause(0.001)
            else:
                print(f"Pos: {pose.pos.round(2)}, Euler: {pose.rot_euler.round(1)}, "
                      f"Grip: {pose.open_degree:.1f} | FPS: {ema_fps:.1f}")

    tracker.cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second")
    # on-by-default joint-space test; pass --no-joint to disable
    parser.add_argument("--no-joint", dest="test_joint", action="store_false",
                        help="Use pose-space tracker instead of joint-space")
    parser.add_argument("--model", type=str, default="wilor", help="Hand tracking model to use")

    parser.set_defaults(test_joint=True)
    args = parser.parse_args()

    main(quiet=args.quiet, fps=args.fps, test_joint=args.test_joint,  model=args.model)