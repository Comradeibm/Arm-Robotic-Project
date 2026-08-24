"""
kinematics.py
=============
Forward and Inverse Kinematics for a generic 6-axis (6-DOF) robotic arm,
defined using standard Denavit-Hartenberg (DH) parameters.

Forward Kinematics (FK):
    Given 6 joint angles -> compute the position/orientation of every link
    and the end-effector, via the product of DH transformation matrices.

Inverse Kinematics (IK):
    Given a target end-effector pose (XYZ + orientation) -> compute the
    joint angles that achieve it, using the numerical Jacobian /
    Damped Least Squares (Levenberg-Marquardt style) method. This is
    used instead of a closed-form solution because it generalizes to
    ANY DH parameter set (arbitrary link lengths/offsets), which is what
    a real "6-axis arm" configuration script needs.

Author: Generated for Project 1 - Robotic Arm Kinematics & Path Planning
"""

import numpy as np


class SixAxisArm:
    """
    Represents a 6-DOF revolute robotic arm using standard DH parameters.

    Each row of `dh_params` is [theta_offset, d, a, alpha] where:
        theta_offset : fixed offset added to the joint variable (rad)
        d            : link offset along previous z (m)
        a            : link length along common normal (m)
        alpha        : link twist (rad)

    The joint variable theta_i (the value we solve for) is added to
    theta_offset for revolute joints.
    """

    def __init__(self, dh_params=None, joint_limits=None, link_radius=0.05):
        if dh_params is None:
            # A reasonably-proportioned anthropomorphic 6-DOF arm,
            # loosely similar in scale to a UR5 / small industrial arm.
            # [theta_offset, d, a, alpha]
            self.dh_params = np.array([
                [0.0,   0.15,  0.00,  np.pi / 2],   # J1: base yaw
                [0.0,   0.00,  0.30,  0.0],          # J2: shoulder
                [0.0,   0.00,  0.25,  0.0],          # J3: elbow
                [0.0,   0.20,  0.00,  np.pi / 2],    # J4: wrist roll
                [0.0,   0.00,  0.00, -np.pi / 2],    # J5: wrist pitch
                [0.0,   0.10,  0.00,  0.0],          # J6: wrist yaw / flange
            ], dtype=float)
        else:
            self.dh_params = np.array(dh_params, dtype=float)

        self.n_joints = self.dh_params.shape[0]

        if joint_limits is None:
            # Default: +-170 deg on all joints (typical industrial arm)
            lim = np.deg2rad(170)
            self.joint_limits = np.array([[-lim, lim]] * self.n_joints)
        else:
            self.joint_limits = np.array(joint_limits, dtype=float)

        # Approximate radius of the arm's links, used for collision checking.
        self.link_radius = link_radius

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------
    @staticmethod
    def _dh_matrix(theta, d, a, alpha):
        """Standard DH homogeneous transform for one joint."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1],
        ])

    def forward_kinematics(self, joint_angles):
        """
        Compute the full chain of transforms.

        Returns:
            joint_positions: (n_joints+1, 3) XYZ of base + each joint origin
            T_end: 4x4 homogeneous transform of the end-effector
            all_T: list of cumulative 4x4 transforms (one per joint)
        """
        joint_angles = np.asarray(joint_angles, dtype=float)
        assert joint_angles.shape[0] == self.n_joints

        T = np.eye(4)
        positions = [T[:3, 3].copy()]
        all_T = []
        for i in range(self.n_joints):
            theta_off, d, a, alpha = self.dh_params[i]
            theta = theta_off + joint_angles[i]
            Ti = self._dh_matrix(theta, d, a, alpha)
            T = T @ Ti
            positions.append(T[:3, 3].copy())
            all_T.append(T.copy())

        return np.array(positions), T, all_T

    def end_effector_pose(self, joint_angles):
        """Return (xyz, rotation_matrix) of the end-effector."""
        _, T_end, _ = self.forward_kinematics(joint_angles)
        return T_end[:3, 3], T_end[:3, :3]

    # ------------------------------------------------------------------
    # Jacobian (numerical) - used by IK solver
    # ------------------------------------------------------------------
    def _geometric_jacobian(self, joint_angles):
        """
        6xN geometric Jacobian (linear velocity rows 0-2, angular rows 3-5)
        computed from the FK chain (standard revolute-joint formula):
            Jv_i = z_{i-1} x (p_end - p_{i-1})
            Jw_i = z_{i-1}
        """
        positions, T_end, all_T = self.forward_kinematics(joint_angles)
        p_end = T_end[:3, 3]

        J = np.zeros((6, self.n_joints))
        z_prev = np.array([0.0, 0.0, 1.0])  # base frame z-axis
        p_prev = positions[0]

        for i in range(self.n_joints):
            Jv = np.cross(z_prev, p_end - p_prev)
            J[0:3, i] = Jv
            J[3:6, i] = z_prev
            # advance to this joint's frame for the next iteration
            z_prev = all_T[i][:3, 2]
            p_prev = all_T[i][:3, 3]
        return J

    # ------------------------------------------------------------------
    # Inverse Kinematics (Damped Least Squares / Levenberg-Marquardt)
    # ------------------------------------------------------------------
    def inverse_kinematics(self, target_xyz, target_R=None, q_init=None,
                            max_iters=300, tol_pos=1e-4, tol_rot=1e-3,
                            damping=0.02, step_clamp=0.15):
        """
        Solve for joint angles that place the end-effector at target_xyz
        (and optionally with orientation target_R, a 3x3 rotation matrix).

        Uses damped least squares: dq = J^T (J J^T + damping^2 I)^-1 * e
        which is numerically stable near singularities (unlike a plain
        pseudo-inverse), and is the standard method for general (non
        closed-form-friendly) DH chains.

        Returns:
            q (np.ndarray): solved joint angles (clipped to joint limits)
            success (bool): whether tolerance was reached
            iters (int): iterations used
        """
        if q_init is None:
            q = np.zeros(self.n_joints)
        else:
            q = np.array(q_init, dtype=float)

        target_xyz = np.asarray(target_xyz, dtype=float)
        use_orientation = target_R is not None

        for it in range(max_iters):
            xyz, R = self.end_effector_pose(q)
            pos_err = target_xyz - xyz

            if use_orientation:
                R_err = target_R @ R.T
                # axis-angle error vector from rotation matrix
                rot_err = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
                err = np.concatenate([pos_err, rot_err])
            else:
                err = np.concatenate([pos_err, np.zeros(3)])

            if np.linalg.norm(pos_err) < tol_pos and (
                not use_orientation or np.linalg.norm(rot_err) < tol_rot
            ):
                return self._clip_to_limits(q), True, it

            J = self._geometric_jacobian(q)
            if not use_orientation:
                J = J.copy()
                J[3:6, :] = 0.0  # ignore orientation rows if not needed

            JJt = J @ J.T
            lam2I = (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt + lam2I, err)

            # clamp step size for stability
            step_norm = np.linalg.norm(dq)
            if step_norm > step_clamp:
                dq = dq * (step_clamp / step_norm)

            q = q + dq
            q = self._clip_to_limits(q)

        return q, False, max_iters

    def _clip_to_limits(self, q):
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def is_within_limits(self, q):
        return bool(np.all(q >= self.joint_limits[:, 0]) and
                    np.all(q <= self.joint_limits[:, 1]))


if __name__ == "__main__":
    # Quick sanity check: FK then IK back to the same point should converge.
    arm = SixAxisArm()
    q_true = np.deg2rad([20, -30, 45, 10, -20, 5])
    xyz, R = arm.end_effector_pose(q_true)
    print("Target XYZ from FK:", xyz)

    q_solved, ok, iters = arm.inverse_kinematics(xyz, q_init=np.zeros(6))
    xyz_check, _ = arm.end_effector_pose(q_solved)
    print(f"IK converged={ok} in {iters} iters, solved XYZ: {xyz_check}")
    print("Position error (m):", np.linalg.norm(xyz - xyz_check))
