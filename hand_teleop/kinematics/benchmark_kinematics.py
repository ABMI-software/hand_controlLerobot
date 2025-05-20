"""
speed_test.py – quick-n-dirty speed test for FK/IK with debug printing
$ python speed_test.py
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_teleop.kinematics.kinematics import RobotKinematics


def make_robot(urdf: str):
    return RobotKinematics(
        urdf_path=urdf,
        frame_name="gripper_link",
    )


def random_deg(n: int, low=-90, high=90):
    return np.random.uniform(low, high, size=n)


def print_pose_info(T, label):
    pos = T[:3, 3]
    euler = R.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)
    print(f"{label} Position: {np.round(pos, 5)} | Euler angles (ZYX, deg): {np.round(euler, 2)}")


def bench_fk(robot_name: str, n_iter=10_000):
    robot = make_robot(robot_name)

    nq = robot.model.nq if robot.model.nq is not None else 6
    q_rad = np.deg2rad(random_deg(nq))

    # Warm-up
    robot.fk(q_rad)

    # Timing
    t0 = time.perf_counter()
    for _ in range(n_iter):
        robot.fk(q_rad)
    t_elapsed = time.perf_counter() - t0

    print(f"FK [{robot_name:5}]:  pinocchio {1e6 * t_elapsed / n_iter:6.1f} µs")


def bench_ik(robot_name: str, n_iter=200):
    robot = make_robot(robot_name)

    nq = robot.model.nq
    goal_deg = np.array([10, 20, 30, -10, 5])
    goal_T = robot.fk(np.deg2rad(goal_deg))
    q0 = np.zeros(nq)

    # Warm-up
    robot.ik(q0.copy(), goal_T)

    # Timing
    t0 = time.perf_counter()
    for _ in range(n_iter):
        robot.ik(q0.copy(), goal_T)
    t_elapsed = time.perf_counter() - t0

    print(f"IK [{robot_name:5}]:  pinocchio {1e3 * t_elapsed / n_iter:6.2f} ms")


def debug_fk(robot_name: str):
    urdf_path = f"{robot_name}.urdf"
    robot = make_robot(urdf_path)

    nq = robot.model.nq if robot.model.nq is not None else 6

    print("=== FK zero position ===")
    T = robot.fk(np.zeros(nq))
    print_pose_info(T, "Zero")

    print("\n=== FK single joint test ===")
    q = np.zeros(nq)
    q[0] = np.deg2rad(30)
    T = robot.fk(q)
    print_pose_info(T, "Shoulder 30°")

    print("\n=== FK random pose ===")
    q = np.deg2rad(random_deg(nq))
    T = robot.fk(q)
    print_pose_info(T, "Random")


if __name__ == "__main__":
    name = "so100"
    print("\n=== FK/IK Benchmarks ===\n")
    bench_fk(name)
    bench_ik(name)
    # debug_fk(name)  # Uncomment to test poses
