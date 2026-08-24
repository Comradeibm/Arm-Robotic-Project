"""
trajectory_planning.py
=======================
Turns a list of joint-space waypoints (e.g. the shortcut RRT path) into a
smooth, time-parameterized trajectory suitable for sending to a real
motion controller (or a ROS `JointTrajectory` message).

Method: per-joint quintic (5th-order) polynomial spline, piecewise
between waypoints, matching position/velocity/acceleration continuity
at interior waypoints, and zero velocity/acceleration at the start and
end (a standard "trapezoidal-smooth" motion profile requirement so the
arm starts and stops gently instead of jerking).

We build this with scipy's `CubicSpline`... actually we go one better
and use scipy's `BPoly.from_derivatives` under the hood via a
finite-difference velocity estimate at interior points, then fit a
quintic on each segment. This keeps the implementation dependency-light
(numpy + scipy.interpolate only) while giving C2-smooth motion.
"""

import numpy as np
from scipy.interpolate import BPoly


class TrajectoryPlanner:
    def __init__(self, waypoints, total_time=None, time_per_segment=1.5):
        """
        waypoints: list/array of shape (n_waypoints, n_joints) - joint
                   angles the arm must pass through, in order.
        total_time: if given, waypoints are spread evenly across this
                    total duration (seconds). Otherwise each segment
                    gets `time_per_segment` seconds.
        """
        self.waypoints = np.array(waypoints, dtype=float)
        self.n_wp, self.n_joints = self.waypoints.shape

        if self.n_wp < 2:
            raise ValueError("Need at least 2 waypoints to build a trajectory.")

        if total_time is not None:
            self.times = np.linspace(0.0, total_time, self.n_wp)
        else:
            self.times = np.arange(self.n_wp) * time_per_segment

        self.splines = self._build_splines()

    def _build_splines(self):
        """
        One BPoly (piecewise Bernstein/quintic-capable polynomial) per
        joint, built from position + estimated velocity + zero
        acceleration at every waypoint (endpoints get zero velocity too).
        """
        splines = []
        for j in range(self.n_joints):
            pos = self.waypoints[:, j]

            # Estimate velocities at interior points via central
            # differences (standard "natural" spline velocity estimate);
            # zero velocity at the first/last waypoint (start/stop at rest).
            vel = np.zeros(self.n_wp)
            for i in range(1, self.n_wp - 1):
                dt_prev = self.times[i] - self.times[i - 1]
                dt_next = self.times[i + 1] - self.times[i]
                v_prev = (pos[i] - pos[i - 1]) / dt_prev
                v_next = (pos[i + 1] - pos[i]) / dt_next
                # zero-crossing check avoids overshoot at local extrema
                vel[i] = 0.0 if v_prev * v_next < 0 else 0.5 * (v_prev + v_next)

            acc = np.zeros(self.n_wp)  # zero accel at every waypoint -> quintic segments

            # BPoly.from_derivatives wants, per waypoint, the list of
            # derivatives [pos, vel, acc, ...] -> quintic (degree 5) needs
            # 3 derivatives (pos,vel,acc) matched at both ends of each
            # segment (6 constraints = degree-5 polynomial).
            yi = [[pos[i], vel[i], acc[i]] for i in range(self.n_wp)]
            spline = BPoly.from_derivatives(self.times, yi, orders=5)
            splines.append(spline)
        return splines

    def sample(self, n_samples=200):
        """
        Returns:
            t: (n_samples,) time vector
            q: (n_samples, n_joints) joint positions
            qd: (n_samples, n_joints) joint velocities
            qdd: (n_samples, n_joints) joint accelerations
        """
        t = np.linspace(self.times[0], self.times[-1], n_samples)
        q = np.zeros((n_samples, self.n_joints))
        qd = np.zeros((n_samples, self.n_joints))
        qdd = np.zeros((n_samples, self.n_joints))
        for j, spline in enumerate(self.splines):
            q[:, j] = spline(t)
            qd[:, j] = spline.derivative(1)(t)
            qdd[:, j] = spline.derivative(2)(t)
        return t, q, qd, qdd

    def total_duration(self):
        return float(self.times[-1] - self.times[0])


if __name__ == "__main__":
    # Quick check: a 3-waypoint trajectory should start/end at rest
    # (zero velocity) and pass through the middle waypoint smoothly.
    wps = np.array([
        [0, 0, 0, 0, 0, 0],
        [0.5, -0.3, 0.4, 0.1, -0.2, 0.0],
        [1.0, -0.1, 0.2, 0.0, 0.1, 0.3],
    ])
    planner = TrajectoryPlanner(wps, time_per_segment=2.0)
    t, q, qd, qdd = planner.sample(50)
    print("Start velocity (should be ~0):", qd[0])
    print("End velocity (should be ~0):", qd[-1])
    print("Trajectory duration (s):", planner.total_duration())
