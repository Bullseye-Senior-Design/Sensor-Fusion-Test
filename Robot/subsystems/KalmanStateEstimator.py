"""Extended Kalman Filter for 9-axis IMU (accel, gyro, mag) and UWB ranges.

State vector (15): [pos(3), vel(3), quat(4), ba(3), bg(3)]
 - pos: x, y, z
 - vel: vx, vy, vz
 - quat: qx, qy, qz, qw (unit quaternion)
 - ba: accel bias
 - bg: gyro bias

This is a minimal, well-documented implementation suitable for simulation and small robots.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
import threading
from dataclasses import dataclass
from typing import Tuple


GRAVITY = np.array([0.0, 0.0, -9.80665])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    # q = [qx, qy, qz, qw]
    qx, qy, qz, qw = q
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, q = q1 * q2
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
    # dtheta: 3-vector small rotation
    theta = np.linalg.norm(dtheta)
    if theta < 1e-8:
        q = np.concatenate((0.5 * dtheta, np.array([1.0])))
    else:
        axis = dtheta / theta
        s = np.sin(theta / 2.0)
        q = np.concatenate((axis * s, np.array([np.cos(theta / 2.0)])))
    return quat_normalize(q)

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    qx, qy, qz, qw = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qx, qy, qz, qw].

    Assumes intrinsic rotations about x (roll), y (pitch), z (yaw) with the
    same convention used by quat_to_euler.
    """
    roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([qx, qy, qz, qw], dtype=float))

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
        # State: pos(3), vel(3), quat(4), ba(3), bg(3) => 16 but quat normalized -> 15 DOF
        # We'll store as a vector of length 16 for convenience but treat quat specially
        self.x = np.zeros(16)
        # init quat as identity
        self.x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])

        # Covariance
        P_pos = np.eye(3) * 1e-2
        P_vel = np.eye(3) * 1e-2
        P_quat = np.eye(3) * 1e-3  # minimal attitude error representation (3x3)
        P_ba = np.eye(3) * 1e-4
        P_bg = np.eye(3) * 1e-4

        # We represent full P as 15x15 matching error-state [pos, vel, att_err(3), ba, bg]
        self.P = np.zeros((15, 15))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.P[6:9, 6:9] = P_quat
        # bias covariances
        self.P[9:12, 9:12] = P_ba
        self.P[12:15, 12:15] = P_bg

        # Process noise (continuous) in error-state
        q_pos = 1e-3
        q_vel = 1e-2
        q_att = 1e-6
        q_ba = 1e-6
        q_bg = 1e-6
        self.Qc = block_diag(np.eye(3) * q_pos, np.eye(3) * q_vel, np.eye(3) * q_att, np.eye(3) * q_ba, np.eye(3) * q_bg)

        # Measurement noise templates
        self.R_uwb_range = 0.1 ** 2  # 10 cm sigma default
        self.R_mag = np.eye(3) * (0.3 ** 2) / 12  # 1 µT std (adjust up/down)

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
        return quat_to_euler(q)

    @property
    def ba(self) -> np.ndarray:
        with self._lock:
            return self.x[10:13].copy()

    @property
    def bg(self) -> np.ndarray:
        with self._lock:
            return self.x[13:16].copy()

    # --- Thread-safe setters for biases
    def set_biases(self, ba: np.ndarray, bg: np.ndarray) -> None:
        """Thread-safe set of accel bias (ba) and gyro bias (bg).

        ba, bg: 3-element array-like each. Stores into internal full-state layout:
          - ba -> self.x[10:13]
          - bg -> self.x[13:16]
        """
        ba_arr = np.asarray(ba, dtype=float).flatten()
        bg_arr = np.asarray(bg, dtype=float).flatten()
        if ba_arr.shape != (3,) or bg_arr.shape != (3,):
            raise ValueError("ba and bg must be 3-element vectors")
        with self._lock:
            self.x[10:13] = ba_arr
            self.x[13:16] = bg_arr

    def set_accel_bias(self, ba: np.ndarray) -> None:
        """Set accel bias (ba) only."""
        ba_arr = np.asarray(ba, dtype=float).flatten()
        if ba_arr.shape != (3,):
            raise ValueError("ba must be a 3-element vector")
        with self._lock:
            self.x[10:13] = ba_arr

    def set_gyro_bias(self, bg: np.ndarray) -> None:
        """Set gyro bias (bg) only."""
        bg_arr = np.asarray(bg, dtype=float).flatten()
        if bg_arr.shape != (3,):
            raise ValueError("bg must be a 3-element vector")
        with self._lock:
            self.x[13:16] = bg_arr
    
    def get_state(self) -> State:
        with self._lock:
            return State(
                pos=tuple(self.x[0:3]),
                vel=tuple(self.x[3:6]),
                quat=tuple(self.x[6:10]),
            )

    def update_uwb_range(self, tag_pos_meas: np.ndarray, tag_offset: np.ndarray | None = None):
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

            o_b = np.zeros(3) if tag_offset is None else np.asarray(tag_offset, dtype=float).reshape(3)

            # rotation matrix from body to world
            q = quat_normalize(self.quat)
            R = quat_to_rotmat(q)

            # prediction h(x)
            h = self.pos + R @ o_b
            y = z - h

            # Jacobian H (3x15): [ I3  03  R[o]_x  03  03 ]
            H = np.zeros((3, 15))
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

            # Joseph form for numerical stability
            I = np.eye(15)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def _inject_error_state(self, dx: np.ndarray):
        """Inject 15-vector error state into the full state x and renormalize quaternion.

        dx layout: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]
        This added instrumentation prints large dx and protects against NaNs.
        """
        if dx.shape[0] != 15:
            raise ValueError("dx must be length 15")
        with self._lock:
            # Diagnostics
            if not np.all(np.isfinite(dx)):
                print(f"[KF _inject_error_state] ERROR: dx contains non-finite values: {dx}")
                # sanitize: replace non-finite with zeros
                dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

            dp = dx[0:3]
            dv = dx[3:6]
            dtheta = dx[6:9]
            dba = dx[9:12]
            dbg = dx[12:15]

            # if np.linalg.norm(dx) > 1.0:
            #     print(f"[KF _inject_error_state] Large injection: |dx|={np.linalg.norm(dx):.3f} dp={dp} dv={dv} dtheta={dtheta}")

            # pos
            self.x[0:3] = self.pos + dp
            # vel
            self.x[3:6] = self.vel + dv
            # attitude: apply small-angle
            dq = small_angle_quat(dtheta)
            q = self.quat
            q_new = quat_mul(dq, q)
            q_new = quat_normalize(q_new)
            self.x[6:10] = q_new
            # accel bias
            self.x[10:13] = self.ba + dba
            # gyro bias
            self.x[13:16] = self.bg + dbg

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
            R_meas: 3x3 measurement covariance in rad^2. If None, defaults to
                    diag([sigma^2, sigma^2, sigma^2]) with sigma = 0.05 rad (~2.9 deg).
        """
        if q_meas is None:
            return

        with self._lock:
            q_est = quat_normalize(self.quat)

            # quaternion conjugate (inverse for unit quaternion)
            q_conj = np.array([-q_est[0], -q_est[1], -q_est[2], q_est[3]], dtype=float)
            q_err = quat_mul(q_meas, q_conj)
            # Make scalar part positive to keep smallest rotation
            if q_err[3] < 0:
                q_err = -q_err

            # small-angle residual (3,)
            y = 2.0 * q_err[0:3]
            if not np.all(np.isfinite(y)):
                return

            # H selects the attitude error block
            H = np.zeros((3, 15))
            H[:, 6:9] = np.eye(3)

            sigma = 0.05  # rad (~2.9 deg)
            R_meas = np.eye(3) * (sigma ** 2)

            S = H @ self.P @ H.T + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return
            K = self.P @ H.T @ Sinv

            dx = (K @ y).flatten()
            self._inject_error_state(dx)

            # Joseph form for numerical stability
            I = np.eye(15)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T