"""
rrt_planner.py
==============
Collision-free path planning in JOINT SPACE (6-D configuration space)
using Rapidly-exploring Random Trees (RRT).

Why joint space, and why RRT?
    - Collision constraints are geometric and non-convex in general, so a
      straight interpolation between the start and goal joint angles can
      pass straight through an obstacle even though both endpoints are
      valid. RRT explores the free configuration space and is a standard,
      simple, and reliable way to find a collision-free path without
      needing an explicit map of "free space" in closed form.
    - Planning in joint space (rather than Cartesian/task space) means
      every sampled point is automatically a physically valid arm pose
      (no extra IK solve needed per sample), and collision checking via
      FK is cheap.

Pipeline:
    1. RRT finds *a* collision-free path (jagged, many waypoints).
    2. Shortcutting removes redundant waypoints by trying to connect
       non-adjacent waypoints directly when the straight joint-space edge
       is collision-free (classic RRT post-processing step).
    3. The shortened waypoint list is handed to trajectory_planning.py to
       be fit with a smooth (quintic) spline.
"""

import numpy as np


class RRTPlanner:
    def __init__(self, arm, checker, joint_limits=None, step_size=0.25,
                 goal_bias=0.1, max_iters=5000, seed=None):
        self.arm = arm
        self.checker = checker
        self.joint_limits = joint_limits if joint_limits is not None else arm.joint_limits
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)
        self.n = arm.n_joints

    def _random_config(self):
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return low + self.rng.random(self.n) * (high - low)

    @staticmethod
    def _nearest(tree_nodes, q):
        dists = [np.linalg.norm(q - node) for node in tree_nodes]
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def _steer(self, q_near, q_rand):
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_rand
        return q_near + direction * (self.step_size / dist)

    def plan(self, q_start, q_goal, goal_tol=0.05):
        """
        Returns (path, success) where path is a list of joint-angle
        waypoints (np.ndarray) from q_start to q_goal, or (None, False)
        if no path was found within max_iters.
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        if self.checker.is_colliding(q_start):
            raise ValueError("Start configuration is in collision.")
        if self.checker.is_colliding(q_goal):
            raise ValueError("Goal configuration is in collision.")

        nodes = [q_start]
        parents = [-1]

        for _ in range(self.max_iters):
            if self.rng.random() < self.goal_bias:
                q_rand = q_goal
            else:
                q_rand = self._random_config()

            idx_near, _ = self._nearest(nodes, q_rand)
            q_near = nodes[idx_near]
            q_new = self._steer(q_near, q_rand)

            if self.checker.path_is_colliding(q_near, q_new, n_substeps=8):
                continue

            nodes.append(q_new)
            parents.append(idx_near)

            if np.linalg.norm(q_new - q_goal) < goal_tol:
                if not self.checker.path_is_colliding(q_new, q_goal, n_substeps=8):
                    nodes.append(q_goal)
                    parents.append(len(nodes) - 2)
                    return self._extract_path(nodes, parents), True

        return None, False

    @staticmethod
    def _extract_path(nodes, parents):
        path = []
        idx = len(nodes) - 1
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path

    def shortcut(self, path, n_attempts=200):
        """
        Randomly try connecting non-adjacent waypoints directly; if the
        direct edge is collision-free, cut out everything in between.
        Reduces a jagged RRT path down to a small number of waypoints,
        which produces a much smoother final spline.
        """
        path = list(path)
        for _ in range(n_attempts):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), size=2, replace=False))
            if j - i < 2:
                continue
            if not self.checker.path_is_colliding(path[i], path[j], n_substeps=10):
                path = path[:i + 1] + path[j:]
        return path


if __name__ == "__main__":
    from kinematics import SixAxisArm
    from collision import CollisionChecker

    arm = SixAxisArm()
    checker = CollisionChecker(arm, ground_z=-0.05)
    checker.add_obstacle(center=[0.35, 0.0, 0.15], radius=0.08)

    q_start = np.deg2rad([90, -20, 20, 0, 0, 0])
    q_goal_xyz = [0.35, 0.35, 0.15]

    # Try several IK seeds and keep the first collision-free solution
    # (a real IK problem can have multiple elbow-up/elbow-down solutions;
    # not all of them are collision-free even if the target point is).
    q_goal = None
    seeds = [q_start, np.zeros(6), np.deg2rad([-90, 20, -20, 0, 0, 0]),
             np.deg2rad([45, -45, 45, 0, 45, 0]), np.deg2rad([90, 0, 0, 0, 0, 0])]
    for seed_q in seeds:
        cand, ok, _ = arm.inverse_kinematics(q_goal_xyz, q_init=seed_q)
        if ok and not checker.is_colliding(cand):
            q_goal = cand
            break
    print("IK for goal converged (collision-free):", q_goal is not None)
    if q_goal is None:
        raise SystemExit("No collision-free IK solution found for demo target; adjust target/obstacle.")

    planner = RRTPlanner(arm, checker, seed=42)
    path, found = planner.plan(q_start, q_goal)
    print("RRT found path:", found, "raw waypoints:", len(path) if path else 0)

    if found:
        short_path = planner.shortcut(path)
        print("Shortcut waypoints:", len(short_path))
