"""
demo.py
=======
End-to-end demonstration of the full pipeline for Project 1:

    1. Define the 6-axis arm (DH parameters) and the workspace obstacles.
    2. Solve Inverse Kinematics for a target XYZ (Point B), trying
       multiple seeds to get a collision-free elbow configuration.
    3. Plan a collision-free path in joint space from Point A to Point B
       using RRT, then shorten it.
    4. Fit a smooth quintic trajectory spline through the shortened path.
    5. Verify the *entire sampled trajectory* is collision-free (not just
       the waypoints) and report max joint velocity/acceleration.
    6. Save a static 3D plot and (if possible) an animated GIF showing
       the arm moving from A to B around the obstacle, plus a CSV of the
       full joint trajectory (this CSV is exactly what the ROS
       trajectory_publisher_node.py script sends to the robot).

Run:
    python3 demo.py
Outputs (written to ../output/):
    arm_path_3d.png        - static plot of start/goal pose + swept path + obstacle
    joint_trajectory.png   - joint angle / velocity / acceleration curves
    trajectory.csv         - time, q1..q6, qd1..qd6 for the ROS publisher node
    arm_motion.gif         - (optional) animation of the arm executing the path
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinematics import SixAxisArm
from collision import CollisionChecker
from rrt_planner import RRTPlanner
from trajectory_planning import TrajectoryPlanner

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)


def solve_ik_collision_free(arm, checker, target_xyz, seeds):
    for seed_q in seeds:
        cand, ok, _ = arm.inverse_kinematics(target_xyz, q_init=seed_q)
        if ok and not checker.is_colliding(cand):
            return cand
    return None


def main():
    # ------------------------------------------------------------------
    # 1. Arm + environment setup
    # ------------------------------------------------------------------
    arm = SixAxisArm()
    checker = CollisionChecker(arm, ground_z=-0.05)
    checker.add_obstacle(center=[0.30, 0.15, 0.15], radius=0.09)

    q_A = np.deg2rad([90, -20, 20, 0, 0, 0])   # Point A: joint-space start pose
    point_B_xyz = [0.35, 0.35, 0.15]            # Point B: target end-effector XYZ

    seeds = [q_A, np.zeros(6), np.deg2rad([-90, 20, -20, 0, 0, 0]),
              np.deg2rad([45, -45, 45, 0, 45, 0]), np.deg2rad([90, 0, 0, 0, 0, 0])]
    q_B = solve_ik_collision_free(arm, checker, point_B_xyz, seeds)
    if q_B is None:
        raise RuntimeError("No collision-free IK solution found for Point B.")

    xyz_A, _ = arm.end_effector_pose(q_A)
    xyz_B, _ = arm.end_effector_pose(q_B)
    print(f"Point A (XYZ): {xyz_A}")
    print(f"Point B (XYZ): {xyz_B}  (target requested: {point_B_xyz})")

    # ------------------------------------------------------------------
    # 2. Collision-free path planning (RRT + shortcutting)
    # ------------------------------------------------------------------
    planner = RRTPlanner(arm, checker, step_size=0.25, goal_bias=0.15,
                          max_iters=5000, seed=7)
    raw_path, found = planner.plan(q_A, q_B, goal_tol=0.05)
    if not found:
        raise RuntimeError("RRT failed to find a collision-free path.")
    print(f"RRT raw path waypoints: {len(raw_path)}")

    short_path = planner.shortcut(raw_path, n_attempts=300)
    print(f"Shortcut path waypoints: {len(short_path)}")

    # ------------------------------------------------------------------
    # 3. Smooth trajectory generation (quintic spline through waypoints)
    # ------------------------------------------------------------------
    traj = TrajectoryPlanner(short_path, time_per_segment=2.0)
    t, q, qd, qdd = traj.sample(n_samples=300)

    # ------------------------------------------------------------------
    # 4. Verify the *sampled* trajectory (not just waypoints) is
    #    collision-free -- the spline can, in principle, bulge slightly
    #    off the waypoint path, so this is a required safety check.
    # ------------------------------------------------------------------
    collisions = [i for i in range(len(t)) if checker.is_colliding(q[i])]
    if collisions:
        print(f"WARNING: {len(collisions)} sampled trajectory points collide! "
              f"Reduce time_per_segment change or add more shortcut waypoints.")
    else:
        print("Verified: full sampled trajectory is collision-free.")

    print(f"Trajectory duration: {traj.total_duration():.2f} s")
    print(f"Max |joint velocity|:     {np.max(np.abs(qd)):.3f} rad/s")
    print(f"Max |joint acceleration|: {np.max(np.abs(qdd)):.3f} rad/s^2")

    # ------------------------------------------------------------------
    # 5. Save trajectory.csv (this is what ros/scripts/trajectory_publisher_node.py
    #    reads and streams to the robot's joint_trajectory_controller)
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "trajectory.csv")
    header = "t," + ",".join(f"q{i+1}" for i in range(6)) + "," + \
             ",".join(f"qd{i+1}" for i in range(6))
    data = np.hstack([t.reshape(-1, 1), q, qd])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    _plot_3d_path(arm, checker, q, xyz_A, xyz_B)
    _plot_joint_curves(t, q, qd, qdd)
    _save_animation(arm, checker, q, xyz_A, xyz_B)


def _arm_segments(arm, q):
    positions, _, _ = arm.forward_kinematics(q)
    return positions


def _plot_3d_path(arm, checker, q_samples, xyz_A, xyz_B):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Obstacle spheres
    for obs in checker.obstacles:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        xs = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        ys = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        zs = obs.center[2] + obs.radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, color="crimson", alpha=0.35, linewidth=0)

    # Swept end-effector path
    ee_path = np.array([_arm_segments(arm, q)[-1] for q in q_samples])
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="royalblue",
            linewidth=2, label="End-effector path")

    # Start and goal arm poses (full skeleton)
    for q, color, label in [(q_samples[0], "seagreen", "Start pose (A)"),
                             (q_samples[-1], "darkorange", "Goal pose (B)")]:
        pts = _arm_segments(arm, q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=4,
                marker="o", markersize=4, label=label)

    ax.scatter(*xyz_A, color="seagreen", s=80, marker="^")
    ax.scatter(*xyz_B, color="darkorange", s=80, marker="^")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("6-Axis Arm: Collision-Free Path from A to B")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "arm_path_3d.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_joint_curves(t, q, qd, qdd):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = [f"J{i+1}" for i in range(q.shape[1])]

    for j in range(q.shape[1]):
        axes[0].plot(t, np.rad2deg(q[:, j]), label=labels[j])
        axes[1].plot(t, np.rad2deg(qd[:, j]), label=labels[j])
        axes[2].plot(t, np.rad2deg(qdd[:, j]), label=labels[j])

    axes[0].set_ylabel("Position (deg)")
    axes[1].set_ylabel("Velocity (deg/s)")
    axes[2].set_ylabel("Acceleration (deg/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("Joint-Space Trajectory (quintic spline)")
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "joint_trajectory.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_animation(arm, checker, q_samples, xyz_A, xyz_B, stride=6):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("matplotlib animation writer unavailable; skipping GIF.")
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for obs in checker.obstacles:
        u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:8j]
        xs = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        ys = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        zs = obs.center[2] + obs.radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, color="crimson", alpha=0.3, linewidth=0)

    ax.scatter(*xyz_A, color="seagreen", s=60, marker="^")
    ax.scatter(*xyz_B, color="darkorange", s=60, marker="^")
    all_pts = np.array([_arm_segments(arm, q) for q in q_samples[::stride]])
    ax.set_xlim(all_pts[:, :, 0].min() - 0.1, all_pts[:, :, 0].max() + 0.1)
    ax.set_ylim(all_pts[:, :, 1].min() - 0.1, all_pts[:, :, 1].max() + 0.1)
    ax.set_zlim(min(0, all_pts[:, :, 2].min() - 0.1), all_pts[:, :, 2].max() + 0.1)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Robotic Arm: A -> B (collision-free)")

    line, = ax.plot([], [], [], color="royalblue", linewidth=4, marker="o", markersize=4)
    frames = q_samples[::stride]

    def update(i):
        pts = _arm_segments(arm, frames[i])
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    out_path = os.path.join(OUT_DIR, "arm_motion.gif")
    try:
        anim.save(out_path, writer=PillowWriter(fps=15))
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Could not save animation GIF ({e}); skipping.")
    plt.close(fig)


if __name__ == "__main__":
    main()
