import argparse
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.tracking.tracker import HandTracker


def main(quiet=False, fps=60):
    tracker = HandTracker(cam_idx=1, hand="right", model="mediapipe")

    follower_pos = np.array([0.2, 0, 0.1])
    follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix()
    follower_vec13 = GripperPose(follower_pos, follower_rot, open_degree=5)

    target_fps = fps
    target_dt = 1.0 / target_fps
    _ema_fps = None
    correction = 0.0  # adaptive correction to sleep

    while tracker.cap.isOpened():
        t0 = time.perf_counter()

        try:
            pose = tracker.read_hand_state(follower_vec13)
        except RuntimeError:
            break

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        # Sleep with adaptive correction
        remain = target_dt - (time.perf_counter() - t0) - correction
        if remain > 0:
            time.sleep(remain)

        # Full frame time
        dt = time.perf_counter() - t0
        fps = 1.0 / dt
        _ema_fps = fps if _ema_fps is None else 0.9 * _ema_fps + 0.1 * fps

        # Update correction based on overshoot
        error = _ema_fps - target_fps
        correction -= 0.0005 * error  # learning rate
        correction = max(0, min(correction, 0.02))  # clamp to [0, 20ms]

        if not quiet:
            print(
                f"{f'Pos: {pose.pos.round(2)}, Euler: {pose.rot_euler.round(1)}, Grip: {pose.open_degree:.1f} | ' if pose is not None else ''}"
                f"FPS: {_ema_fps:.1f}"
            )

    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second")
    args = parser.parse_args()
    main(quiet=args.quiet, fps=args.fps)
