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

    # Replace the existing predict, update_mag and _inject_error_state methods with these.

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray):
        """Predict step using IMU measurements (body frame accel and angular rate).

        accel_meas and gyro_meas are raw sensor readings (3,).
        Added: diagnostics and a safety clamp for excessive world acceleration.
        """
        with self._lock:
            dt = self.dt
            # remove biases (copy biases under lock)
            a = accel_meas - self.ba
            w = gyro_meas - self.bg

            # rotation matrix from body to world
            q = quat_normalize(self.quat)
            R = quat_to_rotmat(q)

            # compute world acceleration
            a_world = R @ a + GRAVITY
            a_world_norm = np.linalg.norm(a_world)

            # Diagnostic: detect huge world acceleration (possible quat/R or accel unit problem)
            if not np.isfinite(a_world_norm) or a_world_norm > 100.0:
                print(f"[KF predict] WARNING: large/invalid world accel |a_world|={a_world_norm:.3e}, a_body={a}, quat={q}")
                # protect against NaNs or extremely large values by clamping
                if not np.isfinite(a_world_norm):
                    # don't use a_world if it's NaN/Inf: fallback to gravity only (stop integrating bogus accel)
                    a_world = GRAVITY.copy()
                else:
                    # clamp magnitude while preserving direction
                    a_world = a_world * (50.0 / a_world_norm)

            # kinematic propagation
            self.x[0:3] = self.pos + self.vel * dt + 0.5 * a_world * dt * dt
            self.x[3:6] = self.vel + a_world * dt

            # attitude update via small-angle approximation
            dq = small_angle_quat(w * dt)
            q_new = quat_mul(dq, q)
            q_new = quat_normalize(q_new)
            self.x[6:10] = q_new

            # biases assumed constant (random walk)

            # Discrete-time process noise
            # Simple linearized F for error-state propagation (15x15)
            F = np.zeros((15, 15))
            # pos dot = vel
            F[0:3, 3:6] = np.eye(3)
            # vel depends on attitude and accel bias
            ax = a
            def skew(v):
                return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

            F[3:6, 6:9] = -R @ skew(ax)
            F[3:6, 9:12] = -R  # partial w.r.t accel bias (in body frame)

            # attitude error rate ~ -I * (w - bg)
            F[6:9, 12:15] = -np.eye(3)

            # Discretize: Phi = I + F*dt
            Phi = np.eye(15) + F * dt

            # Discrete Q
            Qd = Phi @ (self.Qc * dt) @ Phi.T

            # Covariance propagate
            self.P = Phi @ self.P @ Phi.T + Qd

    def update_uwb_range(self, anchor_pos: np.ndarray, range_meas: float, tag_offset: np.ndarray | None = None):
        """UWB range measurement to known anchor position in world frame.

        Args:
            anchor_pos: 3D position of UWB anchor in world frame (3,)
            range_meas: measured range (scalar) from tag to anchor
            tag_offset: 3D offset of UWB tag from robot center in body frame (3,)
                    If None, assumes tag is at robot center
        """
        with self._lock:
            z = range_meas
            
            # Get current rotation matrix to transform body frame to world frame
            q = quat_normalize(self.quat)
            R = quat_to_rotmat(q)
            
            # Calculate tag position in world frame
            # tag_pos_world = robot_pos + R @ tag_offset_body
            if tag_offset is not None:
                tag_pos_world = self.pos + R @ tag_offset
            else:
                tag_pos_world = self.pos
            
            # Predicted range from tag to anchor
            diff = tag_pos_world - anchor_pos
            h = np.linalg.norm(diff)
            if h < 1e-8:
                return
            
            # Measurement Jacobian H (1x15) wrt error-state [pos, vel, att, ba, bg]
            H = np.zeros((1, 15))
            
            # Partial derivative w.r.t position: d(range)/d(pos) = diff / h
            H[0, 0:3] = diff / h
            
            # Partial derivative w.r.t attitude (due to tag offset rotation)
            if tag_offset is not None:
                def skew(v):
                    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                
                # d(tag_pos)/d(theta) = d(R @ tag_offset)/d(theta) = R @ [tag_offset]_x
                # d(range)/d(theta) = (diff/h)^T @ d(tag_pos)/d(theta)
                H[0, 6:9] = (diff / h) @ R @ skew(tag_offset)

            S = H @ self.P @ H.T + self.R_uwb_range
            K = (self.P @ H.T) / S

            y = z - h
            dx = (K * y).flatten()

            # inject and update P under lock
            self._inject_error_state(dx)

            # Joseph form
            self.P = (np.eye(15) - K @ H) @ self.P @ (np.eye(15) - K @ H).T + K * self.R_uwb_range * K.T

    def update_mag(self, mag_meas: np.ndarray, mag_ref: np.ndarray):
        """Magnetometer vector measurement (body frame) with reference vector in world frame.

        We predict mag in body frame: b_body = R.T @ mag_ref_world
        and compare to measured mag_meas.

        Uses direction-only comparison (unit vectors) to avoid magnitude/units mismatch
        between the reference and raw magnetometer units (e.g., µT).
        """
        with self._lock:
            # Normalize measurement to a unit vector (direction only)
            mag_norm = np.linalg.norm(mag_meas)
            if not np.isfinite(mag_norm) or mag_norm < 1e-6:
                # bad sample
                print(f"[KF update_mag] skipped - bad mag sample (norm={mag_norm})")
                return
            mag_meas_u = mag_meas / mag_norm

            q = quat_normalize(self.quat)
            R = quat_to_rotmat(q)
            # predicted body-frame mag (may be any scale) -> normalize for direction
            b_pred = R.T @ mag_ref
            b_pred_norm = np.linalg.norm(b_pred)
            if not np.isfinite(b_pred_norm) or b_pred_norm < 1e-8:
                print(f"[KF update_mag] skipped - predicted mag has invalid norm (b_pred_norm={b_pred_norm})")
                return
            b_pred_u = b_pred / b_pred_norm

            # direction innovation
            y = mag_meas_u - b_pred_u
            y_norm = np.linalg.norm(y)

            # Jacobian approximate using unit reference direction
            H = np.zeros((3, 15))
            def skew(v):
                return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            mag_ref_u = mag_ref / (np.linalg.norm(mag_ref) + 1e-12)
            H[:, 6:9] = -R.T @ skew(mag_ref_u)

            # measurement covariance for unit-vector residuals (tunable)
            # sigma_dir in radians (approx); e.g. 0.05 rad ~ 2.9 degrees
            sigma_dir = 0.05
            R_dir = np.eye(3) * (sigma_dir ** 2)

            S = H @ self.P @ H.T + R_dir

            # Mahalanobis gating
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                print("[KF update_mag] skipped - S inversion failed")
                return
            # d2 = float(y.T @ Sinv @ y)
            # CHI2_3D_99 = 11.34
            # if d2 > CHI2_3D_99:
            #     # outlier
            #     print(f"[KF update_mag] skipped - innovation gated (d2={d2:.3f}, |y|={y_norm:.3f})")
            #     return

            K = self.P @ H.T @ Sinv
            dx = (K @ y).flatten()

            # sanitize and apply
            if not np.all(np.isfinite(dx)):
                print(f"[KF update_mag] sanitized non-finite dx")
                dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

            self._inject_error_state(dx)

            # covariance update
            self.P = (np.eye(15) - K @ H) @ self.P

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

            if np.linalg.norm(dx) > 1.0:
                print(f"[KF _inject_error_state] Large injection: |dx|={np.linalg.norm(dx):.3f} dp={dp} dv={dv} dtheta={dtheta}")

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