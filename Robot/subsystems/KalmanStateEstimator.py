"""Extended Kalman Filter for 9-axis IMU (accel, gyro, mag) and UWB ranges.

Trimmed variant: state vector (10) = [pos(3), vel(3), quat(4)].
- pos: x, y, z
- vel: vx, vy, vz
- quat: qx, qy, qz, qw (unit quaternion)

Error-state (9) = [pos_err(3), vel_err(3), att_err(3)].

This implementation removes accel/gyro bias states from the filter. It is
intended for use when biases are handled outside the filter (e.g., by the IMU
preprocessor) or not modeled.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
import threading
from dataclasses import dataclass
from typing import Tuple
from Robot.MathUtil import MathUtil
import time

GRAVITY = np.array([0.0, 0.0, -9.80665])

@dataclass
class State:
    pos: Tuple[float, float, float]      # shape (3,)
    vel: Tuple[float, float, float]      # shape (3,)
    quat: Tuple[float, float, float, float]  # shape (4,)

class KalmanStateEstimator:
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start()
        return cls._instance
    
    def _start(self, dt: float = 0.01):
        self.dt = dt
        # Reentrant lock to allow safe concurrent access from multiple threads
        self._lock = threading.RLock()

        # Full state: pos(3), vel(3), quat(4) => 10 elements
        self.x = np.zeros(10)
        # init quat as identity
        self.x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])

        # Error-state covariance for [pos(3), vel(3), att_err(3)] => 9x9
        P_pos = np.eye(3) * 1e-0
        P_vel = np.eye(3) * 1e-0
        P_att = np.eye(3) * 1e-0
        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.P[6:9, 6:9] = P_att

        # Process noise (continuous) in error-state (9x9)
        q_pos = 1e-3
        q_vel = 1e-2
        q_att = 1e-2
        self.Qc = block_diag(np.eye(3) * q_pos, np.eye(3) * q_vel, np.eye(3) * q_att)

        # Measurement noise templates
        self.R_uwb_range = 0.1 ** 2  # 10 cm sigma default (variance)
        # IMU attitude measurement noise (small-angle residuals)
        # default sigma ~0.02 rad (~1.15 deg)
        self.imu_attitude_sigma = 0.04
        self.R_imu_attitude = np.eye(3) * (self.imu_attitude_sigma ** 2)
        
        threading.Thread(target=self._run_loop, daemon=True).start()

    # --- Helpers to access parts of the full state
    @property
    def pos(self) -> np.ndarray:
        with self._lock:
            return self.x[0:3].copy()

    @property
    def vel(self) -> np.ndarray:
        with self._lock:
            return self.x[3:6].copy()

    @property
    def quat(self) -> np.ndarray:
        with self._lock:
            return self.x[6:10].copy()
    
    @property
    def euler(self) -> np.ndarray:
        # quat property already locks so just call it
        q = self.quat
        return MathUtil.quat_to_euler(q)
    
    def get_state(self) -> State:
        with self._lock:
            return State(
                pos=tuple(self.x[0:3]),
                vel=tuple(self.x[3:6]),
                quat=tuple(self.x[6:10]),
            )
    
    def _run_loop(self):
        """Background thread to run predict at fixed dt intervals."""
        import time
        next_time = time.time()
        while True:
            next_time += self.dt
            self.predict()
            sleep_duration = next_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    # Replace the existing predict, update_mag and _inject_error_state methods with these.

    def predict(self):
        """Predict step using IMU acceleration (body frame) when available.

        - Reads accel from the IMU singleton (IMU().get_accel()).
        - Converts accel to world frame using current quaternion, subtracts GRAVITY
          to get linear acceleration in world frame.
        - Integrates velocity and position with simple constant-acceleration kinematics.
        - Keeps the existing error-state covariance propagation.

        Additional safety:
        - Reject non-finite, implausibly large or spikey accel samples.
        - After several consecutive rejections, fall back to constant-velocity
          until a valid sample arrives.
        """
        # import IMU here to avoid circular import at module load time
        from Robot.subsystems.sensors.IMU import IMU

        with self._lock:
            dt = self.dt

            # Try to get accelerometer reading from IMU singleton
            accel_tuple = None
            accel_tuple = IMU().get_accel()

            a_world = np.zeros(3)

            if accel_tuple is not None:
                a_b = np.asarray(accel_tuple, dtype=float).reshape(3)
                if np.all(np.isfinite(a_b)):
                    # rotate accel to world frame and remove gravity to get linear acceleration
                    q = MathUtil.quat_normalize(self.quat)
                    R = MathUtil.quat_to_rotmat(q)  # body->world
                    a_w = R @ a_b
                    # subtract gravity (world frame)
                    a_lin = a_w - GRAVITY

                    a_world = a_lin

            # Integrate full state using accel if available, else constant-velocity
            if accel_tuple is not None:
                # kinematic propagation with constant acceleration over dt
                # pos = pos + v*dt + 0.5*a*dt^2
                self.x[0:3] = self.pos + self.vel * dt + 0.5 * a_world * (dt ** 2)
                # vel = vel + a*dt
                self.x[3:6] = self.vel + a_world * dt
            else:
                # fallback: constant velocity (no accel)
                self.x[0:3] = self.pos + self.vel * dt
                self.x[3:6] = self.vel

            # Linearized F for error-state propagation (9x9) with pos <- vel coupling
            F = np.zeros((9, 9))
            F[0:3, 3:6] = np.eye(3)

            # Discretize
            Phi = np.eye(9) + F * dt

            # Discrete Q (use existing continuous Qc)
            Qd = Phi @ (self.Qc * dt) @ Phi.T

            # Covariance propagate
            self.P = Phi @ self.P @ Phi.T + Qd

    def update_uwb_range(self, tag_pos_meas: np.ndarray, tag_offset: np.ndarray | None = None, use_offset: bool = True):
        """EKF update using the fused UWB tag world position (POS), not per-anchor ranges.

        Measurement model:
            z (3x1) = h(x) + v,  h(x) = p + R(q) @ o_b
        where
            p is robot position in world (state),
            R(q) is body->world rotation from quaternion,
            o_b is tag offset in body frame (default 0),
            v ~ N(0, R_meas).

        Args:
            tag_pos_meas: (3,) measured tag position in world frame [m]
            tag_offset:   (3,) tag offset in body frame from robot center [m]; None => [0,0,0]
        """        
        with self._lock:
            z = np.asarray(tag_pos_meas, dtype=float).reshape(3)
            if not np.all(np.isfinite(z)):
                return

            # Decide whether to apply the offset
            o_b = np.zeros(3)
            if use_offset and tag_offset is not None:
                o_b = np.asarray(tag_offset, dtype=float).reshape(3)

            # rotation matrix from body to world
            q = MathUtil.quat_normalize(self.quat)
            R = MathUtil.quat_to_rotmat(q)

            # prediction h(x)
            h = self.pos + R @ o_b
            y = z - h

            # Jacobian H (3x9): [ I3  03  R[o]_x ] wrt error-state [pos, vel, att_err]
            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)

            if np.any(o_b):
                def skew(v):
                    return np.array([[0, -v[2], v[1]],
                                     [v[2], 0, -v[0]],
                                     [-v[1], v[0], 0]])
                H[:, 6:9] = R @ skew(o_b)

            R_meas = np.eye(3) * self.R_uwb_range

            S = H @ self.P @ H.T + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return

            K = self.P @ H.T @ Sinv

            dx = (K @ y).flatten()

            # inject and update P under lock
            self._inject_error_state(dx)

            # Joseph form for numerical stability (9x9)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def _inject_error_state(self, dx: np.ndarray):
        """Inject 9-vector error state into the full state x and renormalize quaternion.

        dx layout: [dp(3), dv(3), dtheta(3)]
        """
        if dx.shape[0] != 9:
            raise ValueError("dx must be length 9")
        with self._lock:
            # sanitize non-finite
            if not np.all(np.isfinite(dx)):
                dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

            dp = dx[0:3]
            dv = dx[3:6]
            dtheta = dx[6:9]

            # pos
            self.x[0:3] = self.pos + dp
            # vel
            self.x[3:6] = self.vel + dv
            # attitude: apply small-angle
            dq = MathUtil.small_angle_quat(dtheta)
            q = self.quat
            q_new = MathUtil.quat_mul(dq, q)
            q_new = MathUtil.quat_normalize(q_new)
            self.x[6:10] = q_new

    def update_imu_attitude(self, q_meas: np.ndarray | None = None):
        """EKF attitude update using an external IMU rotation estimate.

        Provide either a quaternion q_meas = [qx,qy,qz,qw] or Euler angles
        euler_rpy = [roll, pitch, yaw] in radians. The measurement residual is
        the small-angle vector from the quaternion error:

            q_err = q_meas ⊗ conj(q_est)  (ensure qw >= 0)
            y ≈ 2 * q_err.xyz  (3x1)

        The measurement Jacobian for the error-state is simply H = [0 0 I 0 0]
        for the attitude block, making this a direct attitude observation.

        Args:
            q_meas: quaternion [qx,qy,qz,qw] (preferred).
            euler_rpy: roll, pitch, yaw in radians (used if q_meas is None).
        """
        if q_meas is None:
            return

        with self._lock:
            q_est = MathUtil.quat_normalize(self.quat)

            # quaternion conjugate (inverse for unit quaternion)
            q_conj = np.array([-q_est[0], -q_est[1], -q_est[2], q_est[3]], dtype=float)
            q_err = MathUtil.quat_mul(q_meas, q_conj)
            # Make scalar part positive to keep smallest rotation
            if q_err[3] < 0:
                q_err = -q_err

            # small-angle residual (3,)
            y = 2.0 * q_err[0:3]
            if not np.all(np.isfinite(y)):
                return

            # H selects the attitude error block in the 9-element error-state
            # error-state: [pos(0:3), vel(3:6), att_err(6:9)]
            H = np.zeros((3, 9))
            H[:, 6:9] = np.eye(3)

            # Use preconfigured IMU attitude measurement covariance
            R_meas = self.R_imu_attitude

            S = H @ self.P @ H.T + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return
            K = self.P @ H.T @ Sinv

            dx = (K @ y).flatten()
            self._inject_error_state(dx)

            # Joseph form for numerical stability (9x9)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T