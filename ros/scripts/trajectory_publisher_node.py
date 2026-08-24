#!/usr/bin/env python3
"""
trajectory_publisher_node.py
=============================
Reads the smooth trajectory produced offline by src/demo.py
(../../output/trajectory.csv: columns t, q1..q6, qd1..qd6) and streams it
to the ROS `joint_trajectory_controller` as a single
`trajectory_msgs/JointTrajectory` action goal (FollowJointTrajectory),
which is the standard way to drive a simulated (Gazebo) or real 6-DOF
arm's joints through a pre-planned, collision-free, time-parameterized
path.

This is the "last mile" node: everything upstream (IK, RRT planning,
spline smoothing, collision checking) happens offline/in simulation via
the pure-Python pipeline in src/, producing trajectory.csv. This node's
only job is to get that CSV onto the robot.

Build/run (inside your catkin workspace, after `roslaunch ... arm_sim.launch`):
    rosrun six_dof_arm_description trajectory_publisher_node.py \
        _csv_path:=/absolute/path/to/output/trajectory.csv

NOTE: Requires a real ROS environment (rospy, actionlib,
control_msgs, trajectory_msgs) to run; it is not executed in this
sandbox since ROS is not installed here.
"""

import csv
import sys
import os

try:
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError:
    rospy = None

JOINT_NAMES = [f"joint{i+1}" for i in range(6)]


def load_trajectory_csv(csv_path):
    """Parses the CSV written by src/demo.py into a list of
    (t, positions[6], velocities[6]) tuples."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t"])
            pos = [float(row[f"q{i+1}"]) for i in range(6)]
            vel = [float(row[f"qd{i+1}"]) for i in range(6)]
            rows.append((t, pos, vel))
    return rows


def build_goal(rows):
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = JOINT_NAMES

    for t, pos, vel in rows:
        pt = JointTrajectoryPoint()
        pt.positions = pos
        pt.velocities = vel
        pt.time_from_start = rospy.Duration(t)
        goal.trajectory.points.append(pt)

    return goal


def main():
    rospy.init_node("trajectory_publisher_node")
    csv_path = rospy.get_param("~csv_path",
                                os.path.join(os.path.dirname(__file__),
                                             "..", "..", "output", "trajectory.csv"))
    rospy.loginfo(f"Loading trajectory from {csv_path}")
    rows = load_trajectory_csv(csv_path)
    rospy.loginfo(f"Loaded {len(rows)} trajectory points "
                   f"({rows[-1][0]:.2f}s total duration)")

    client = actionlib.SimpleActionClient(
        "/arm_trajectory_controller/follow_joint_trajectory",
        FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for joint_trajectory_controller action server...")
    client.wait_for_server()

    goal = build_goal(rows)
    rospy.loginfo("Sending trajectory goal to controller...")
    client.send_goal(goal)
    client.wait_for_result()
    rospy.loginfo(f"Trajectory execution result: {client.get_result()}")


if __name__ == "__main__":
    if rospy is None:
        print("rospy not available in this environment -- this node is meant "
              "to run inside a ROS workspace. See module docstring.")
        sys.exit(0)
    main()
