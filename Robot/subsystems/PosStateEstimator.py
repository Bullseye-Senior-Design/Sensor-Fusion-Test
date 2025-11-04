"""State estimator that consumes fused position measurements instead of raw IMU/UWB.

This estimator keeps a minimal state: position (3), velocity (3), attitude (quaternion, 4).
It uses a simple linear Kalman update for fused 3D position measurements. Prediction uses
constant-velocity model (no raw IMU required). This is useful when another sensor-fusion
block already provides a fused/filtered position and this node only needs to fuse it
into a local state estimate and provide velocity estimates.

API (minimal):
 - predict(dt): advances state using constant velocity
 - update_fused_position(pos_meas, R=None): incorporate a 3-vector fused position measurement
 - properties: pos, vel, quat, euler

The implementation is intentionally lightweight and thread-safe (RLock).
"""
from __future__ import annotations
import numpy as np
import threading
from dataclasses import dataclass
from typing import Tuple, Optional

GRAVITY = np.array([0.0, 0.0, -9.80665])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


@dataclass
class State:
    pos: Tuple[float, float, float]
    vel: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]


class PosStateEstimator:
    """Estimator driven by fused 3D position measurements.

    This estimator intentionally omits raw IMU processing. If you have IMU
    available and want to use it for prediction, consider using
    `KalmanStateEstimator` instead.
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self._lock = threading.RLock()

        # full state stored compactly: pos(3), vel(3), quat(4) => 10 elements
        self.x = np.zeros(10)
        self.x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        # Error-state covariance for [pos(3), vel(3), att_err(3)] => 9x9
        P_pos = np.eye(3) * 1e-2
        P_vel = np.eye(3) * 1e-2
        P_att = np.eye(3) * 1e-3
        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.P[6:9, 6:9] = P_att

        # process noise (simple discrete form)
        q_pos = 1e-4
        q_vel = 1e-3
        q_att = 1e-6
        self.Q = np.zeros((9, 9))
        self.Q[0:3, 0:3] = np.eye(3) * q_pos
        self.Q[3:6, 3:6] = np.eye(3) * q_vel
        self.Q[6:9, 6:9] = np.eye(3) * q_att

        # default measurement noise for fused position (3x3)
        self.R_fused_pos = np.eye(3) * (0.1 ** 2)  # 10 cm sigma by default

    # --- accessors
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
        return quat_to_euler(self.quat)

    def get_state(self) -> State:
        with self._lock:
            return State(pos=tuple(self.x[0:3]), vel=tuple(self.x[3:6]), quat=tuple(self.x[6:10]))

    # --- core estimator methods
    def predict(self, dt: Optional[float] = None):
        """Predict step using constant velocity model.

        If dt is None, uses the estimator's configured dt.
        """
        with self._lock:
            if dt is None:
                dt = self.dt

            # state propagation
            self.x[0:3] = self.x[0:3] + self.x[3:6] * dt
            # quaternion unchanged in absence of IMU

            # simple linearized F for [pos, vel, att_err]
            F = np.zeros((9, 9))
            F[0:3, 3:6] = np.eye(3)

            Phi = np.eye(9) + F * dt
            Qd = self.Q * dt
            self.P = Phi @ self.P @ Phi.T + Qd

    def update_fused_position(self, pos_meas: np.ndarray, R: Optional[np.ndarray] = None):
        """Update with a fused 3D position measurement in world frame.

        pos_meas: shape (3,)
        R: optional 3x3 measurement covariance. If None, uses default.
        """
        if pos_meas.shape != (3,) and pos_meas.shape != (3,1):
            raise ValueError("pos_meas must be a 3-vector")

        with self._lock:
            z = pos_meas.flatten()
            if R is None:
                R = self.R_fused_pos

            # measurement model h(x) = pos -> H = [I3 0 0] (1x9 per axis)
            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)

            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)

            y = z - self.pos
            dx = K @ y

            # inject error-state: dx layout [dp(3), dv(3), dtheta(3)]
            self._inject_error_state(dx.flatten())

            # covariance update (Joseph form)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def _inject_error_state(self, dx: np.ndarray):
        if dx.shape[0] != 9:
            raise ValueError("dx must be length 9")
        with self._lock:
            # pos
            self.x[0:3] = self.pos + dx[0:3]
            # vel
            self.x[3:6] = self.vel + dx[3:6]
            # attitude: small-angle add to quaternion
            dtheta = dx[6:9]
            dq = self._small_angle_quat(dtheta)
            q = self.quat
            q_new = self._quat_mul(dq, q)
            self.x[6:10] = quat_normalize(q_new)

    # --- small quaternion helpers (copied minimal implementations)
    def _small_angle_quat(self, dtheta: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(dtheta)
        if theta < 1e-8:
            q = np.concatenate((0.5 * dtheta, np.array([1.0])))
        else:
            axis = dtheta / theta
            s = np.sin(theta / 2.0)
            q = np.concatenate((axis * s, np.array([np.cos(theta / 2.0)])))
        return quat_normalize(q)

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return np.array([x, y, z, w])
