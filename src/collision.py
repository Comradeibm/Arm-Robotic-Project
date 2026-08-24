"""
collision.py
============
Simple, fast collision checking for the robotic arm workspace.

Model:
    - Obstacles are represented as spheres (center, radius). This is a
      common simplification in path planning: any real obstacle can be
      approximated (or bounded) by one or more spheres.
    - Each arm link between consecutive joint positions is treated as a
      capsule (a line segment with a radius = arm.link_radius).
    - Collision = minimum distance between a link segment and an obstacle
      sphere's center is less than (link_radius + obstacle_radius).

This gives O(n_links * n_obstacles) collision checks per configuration,
which is fast enough for sampling-based planners (RRT) at interactive
rates.
"""

import numpy as np


class SphereObstacle:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)


def point_segment_distance(p, a, b):
    """Minimum distance from point p to line segment [a, b]."""
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


class CollisionChecker:
    def __init__(self, arm, obstacles=None, ground_z=None):
        """
        arm: SixAxisArm instance (provides forward_kinematics + link_radius)
        obstacles: list of SphereObstacle
        ground_z: optional float; if set, any link point with z < ground_z
                   counts as a collision (simple floor-plane check).
        """
        self.arm = arm
        self.obstacles = obstacles or []
        self.ground_z = ground_z

    def add_obstacle(self, center, radius):
        self.obstacles.append(SphereObstacle(center, radius))

    def is_colliding(self, joint_angles):
        """Returns True if this arm configuration collides with anything."""
        positions, _, _ = self.arm.forward_kinematics(joint_angles)

        # Floor check
        if self.ground_z is not None:
            if np.any(positions[:, 2] < self.ground_z):
                return True

        # Link-vs-obstacle checks
        for i in range(len(positions) - 1):
            a, b = positions[i], positions[i + 1]
            for obs in self.obstacles:
                d = point_segment_distance(obs.center, a, b)
                if d < (self.arm.link_radius + obs.radius):
                    return True
        return False

    def path_is_colliding(self, q_a, q_b, n_substeps=10):
        """
        Check the straight-line joint-space segment between two
        configurations for collisions by sampling n_substeps points along
        it (a standard "edge validity" check used inside RRT).
        """
        q_a = np.asarray(q_a, dtype=float)
        q_b = np.asarray(q_b, dtype=float)
        for s in np.linspace(0.0, 1.0, n_substeps):
            q = q_a + s * (q_b - q_a)
            if self.is_colliding(q):
                return True
        return False


if __name__ == "__main__":
    from kinematics import SixAxisArm

    arm = SixAxisArm()
    checker = CollisionChecker(arm, ground_z=-0.05)
    checker.add_obstacle(center=[0.35, 0.0, 0.15], radius=0.08)

    q_free = np.deg2rad([90, -20, 20, 0, 0, 0])
    q_hit = np.deg2rad([0, -10, 20, 0, 0, 0])

    print("q_free colliding:", checker.is_colliding(q_free))
    print("q_hit colliding:", checker.is_colliding(q_hit))
