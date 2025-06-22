#!/usr/bin/env python3
"""
benchmark_kinematics.py – Compare FK/IK between:
▪ legacy screw-axis kinematics (kinematics_old.py)
▪ new Pinocchio wrapper        (kinematics.py)

Run:
    python benchmark_kinematics.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_teleop.kinematics.kinematics_old import RobotKinematics as OldFK
from hand_teleop.kinematics.kinematics import RobotKinematics as NewFK

ROBOT = "so100"
FRAME = "gripper_link"
DOF = 5  # All meaningful joints

def rand_deg(n, lo=-90, hi=90):
    return np.random.uniform(lo, hi, size=n)

def pose_err_details(Ta, Tb):
    pos_a, pos_b = Ta[:3, 3], Tb[:3, 3]
    dp = pos_b - pos_a
    dist = np.linalg.norm(dp)

    Ra, Rb = R.from_matrix(Ta[:3, :3]), R.from_matrix(Tb[:3, :3])
    Rdiff = Ra.inv() * Rb
    rotvec = Rdiff.as_rotvec()
    angle_deg = np.linalg.norm(rotvec) * 180 / np.pi

    return dist, angle_deg, dp, rotvec * 180 / np.pi

def print_pose(T, label):
    p = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)
    print(f"{label:<8} Pos: {np.round(p, 3)} | RPY ZYX: {np.round(rpy, 2)}")

def bench(label, fn, iters):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--ik-iters", type=int, default=500)
    args = parser.parse_args()

    old = OldFK(robot_type=ROBOT)
    new = NewFK(
        urdf_path=f"hand_teleop/kinematics/urdf/{ROBOT}.urdf",
        frame_name=FRAME
    )

    # Real joint angles
    real_start_deg = np.array([-175.34, 134.56, 237.74, 93.164, -4.043])
    real_lifted_deg = np.array([-173.27, 114.15, 243.74, 96.265, -2.7376])

    real_start_deg6 = np.append(real_start_deg, 0.0)
    real_lifted_deg6 = np.append(real_lifted_deg, 0.0)

    # ─── FK TEST ─────────────────────────────────────────────────────────────
    t_old = bench("fk-old", lambda: old.fk_gripper(real_start_deg6), args.iters)
    t_new = bench("fk-new", lambda: new.fk(np.deg2rad(real_start_deg)), args.iters)
    print(f"FK  old: {t_old * 1e6:7.1f} µs   new: {t_new * 1e6:7.1f} µs")

    T_old = old.fk_gripper(real_start_deg6)
    T_new = new.fk(np.deg2rad(real_start_deg))
    dp, da, dp_vec, da_vec = pose_err_details(T_old, T_new)

    print(f"FK  pose diff: {dp * 1e3:5.2f} mm | {da:5.2f} °")
    if dp > 1e-3 or da > 0.5:
        print("⚠️  FK mismatch detected:")
        print("→ Old:")
        print_pose(T_old, "Old FK")
        print("→ New:")
        print_pose(T_new, "New FK")
        print(f"→ Pos Δ: {np.round(dp_vec * 1000, 2)} mm")
        print(f"→ Rot Δ (axis-angle): {np.round(da_vec, 2)} °")

    print()  # separator

    # ─── IK TEST ─────────────────────────────────────────────────────────────
    goal_pose = old.fk_gripper(real_lifted_deg6)

    ik_old = lambda: old.ik(real_start_deg6.copy(), goal_pose)  # noqa: E731
    ik_new = lambda: new.ik(np.deg2rad(real_start_deg.copy()), goal_pose, max_iters=3)  # noqa: E731

    t_old_ik = bench("ik-old", ik_old, args.ik_iters)
    t_new_ik = bench("ik-new", ik_new, args.ik_iters)
    print(f"IK  old: {t_old_ik * 1e3:6.2f} ms   new: {t_new_ik * 1e3:6.2f} ms")

    q_sol = new.ik(np.deg2rad(real_start_deg.copy()), goal_pose)
    T_back = new.fk(q_sol)
    dp2, da2, dp_vec2, da_vec2 = pose_err_details(goal_pose, T_back)

    print(f"IK  new pose error: {dp2 * 1e3:5.2f} mm | {da2:5.2f} °")
    if dp2 > 1e-3 or da2 > 0.5:
        print("⚠️  IK result does not match target pose:")
        print("→ Target:")
        print_pose(goal_pose, "Target")
        print("→ IK→FK:")
        print_pose(T_back, "Result")
        print(f"→ Pos Δ: {np.round(dp_vec2 * 1000, 2)} mm")
        print(f"→ Rot Δ (axis-angle): {np.round(da_vec2, 2)} °")

if __name__ == "__main__":
    main()