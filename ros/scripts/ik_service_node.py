#!/usr/bin/env python3
"""
ik_service_node.py
===================
Wraps the SixAxisArm inverse-kinematics solver (src/kinematics.py) as a
ROS service, so any other node (a task planner, a GUI, an RViz
interactive marker callback, etc.) can request "give me joint angles for
this XYZ" over the standard ROS service mechanism.

This file is meant to be copied into a catkin package (e.g.
`six_dof_arm_description/scripts/`) alongside a service definition:

    # srv/SolveIK.srv
    geometry_msgs/Point target_xyz
    float64[] seed_joint_angles   # optional, may be empty
    ---
    float64[] joint_angles
    bool success

Build/run (inside your catkin workspace):
    catkin_make
    source devel/setup.bash
    rosrun six_dof_arm_description ik_service_node.py

Call it:
    rosservice call /solve_ik "{target_xyz: {x: 0.35, y: 0.35, z: 0.15}, seed_joint_angles: []}"

NOTE: This script requires a real ROS (rospy) environment to run; it is
provided ready-to-drop-in but is not executed in this sandbox since ROS
is not installed here. The underlying math (kinematics.SixAxisArm) is
identical to, and was verified against, src/kinematics.py.
"""

import sys
import os

# Make the pure-python kinematics module (src/kinematics.py) importable
# regardless of where this script is installed inside the catkin package.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

try:
    import rospy
    from six_dof_arm_description.srv import SolveIK, SolveIKResponse  # generated from srv/SolveIK.srv
except ImportError:
    rospy = None  # allows `python3 -m py_compile` / linting outside ROS

from kinematics import SixAxisArm


class IKServiceNode:
    def __init__(self):
        self.arm = SixAxisArm()
        rospy.init_node("ik_service_node")
        self.service = rospy.Service("solve_ik", SolveIK, self.handle_request)
        rospy.loginfo("ik_service_node ready: call rosservice call /solve_ik ...")

    def handle_request(self, req):
        target = np.array([req.target_xyz.x, req.target_xyz.y, req.target_xyz.z])

        seeds = []
        if len(req.seed_joint_angles) == self.arm.n_joints:
            seeds.append(np.array(req.seed_joint_angles))
        # Always also try a small bank of default seeds, in case the
        # caller's seed leads to a colliding / unreachable elbow config.
        seeds += [np.zeros(6), np.deg2rad([90, -20, 20, 0, 0, 0]),
                  np.deg2rad([-90, 20, -20, 0, 0, 0])]

        for seed in seeds:
            q, ok, _ = self.arm.inverse_kinematics(target, q_init=seed)
            if ok:
                resp = SolveIKResponse()
                resp.joint_angles = q.tolist()
                resp.success = True
                return resp

        resp = SolveIKResponse()
        resp.joint_angles = []
        resp.success = False
        return resp


if __name__ == "__main__":
    if rospy is None:
        print("rospy not available in this environment -- this node is meant "
              "to run inside a ROS workspace. See module docstring.")
        sys.exit(0)
    node = IKServiceNode()
    rospy.spin()
