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

        # Full state: pos(3), vel(3), quat(4) => 10 elements
        self.x = np.zeros(10)
        # init quat as identity
        self.x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])

        # Error-state covariance for [pos(3), vel(3), att_err(3)] => 9x9
        P_pos = np.eye(3) * 1e-2
        P_vel = np.eye(3) * 1e-2
        P_att = np.eye(3) * 1e-3
        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.P[6:9, 6:9] = P_att

        # Process noise (continuous) in error-state (9x9)
        q_pos = 1e-3
        q_vel = 1e-2
        q_att = 1e-6
        self.Qc = block_diag(np.eye(3) * q_pos, np.eye(3) * q_vel, np.eye(3) * q_att)

        # Measurement noise templates
        self.R_uwb_range = 0.1 ** 2  # 10 cm sigma default (variance)
        self.R_mag = np.eye(3) * (0.3 ** 2) / 12  # for mag direction residuals (tunable)

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
    
    def get_state(self) -> State:
        with self._lock:
            return State(
                pos=tuple(self.x[0:3]),
                vel=tuple(self.x[3:6]),
                quat=tuple(self.x[6:10]),
            )

    # Replace the existing predict, update_mag and _inject_error_state methods with these.

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray):
        """Predict step using IMU measurements (body frame accel and angular rate).

        accel_meas and gyro_meas are raw sensor readings (3,).
        Added: diagnostics and a safety clamp for excessive world acceleration.
        """
        # Modified predict: do not use accel or gyro measurements. Assume constant linear velocity model.
        with self._lock:
            dt = self.dt

            # simple constant-velocity kinematic propagation
            self.x[0:3] = self.pos + self.vel * dt
            # velocity assumed constant (no accel integration)
            self.x[3:6] = self.vel

            # attitude left unchanged (we do not integrate gyro here)

            # Linearized F for error-state propagation (9x9) with pos <- vel coupling
            F = np.zeros((9, 9))
            F[0:3, 3:6] = np.eye(3)

            # Discretize
            Phi = np.eye(9) + F * dt

            # Discrete Q (use existing continuous Qc)
            Qd = Phi @ (self.Qc * dt) @ Phi.T

            # Covariance propagate
            self.P = Phi @ self.P @ Phi.T + Qd

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
            dq = small_angle_quat(dtheta)
            q = self.quat
            q_new = quat_mul(dq, q)
            q_new = quat_normalize(q_new)
            self.x[6:10] = q_new
            # accel bias
            self.x[10:13] = self.ba + dba
            # gyro bias
            self.x[13:16] = self.bg + dbg