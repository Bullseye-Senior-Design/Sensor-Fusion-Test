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


class state_estimator:
    def __init__(self, dt: float = 0.01):
        self.dt = dt
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
        self.R_mag = np.eye(3) * (0.05 ** 2)

    # --- Helpers to access parts of the full state
    @property
    def pos(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def quat(self) -> np.ndarray:
        return self.x[6:10]

    @property
    def ba(self) -> np.ndarray:
        return self.x[10:13]

    @property
    def bg(self) -> np.ndarray:
        return self.x[13:16]

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray):
        """Predict step using IMU measurements (body frame accel and angular rate).

        accel_meas and gyro_meas are raw sensor readings (3,).
        """
        dt = self.dt
        # remove biases
        a = accel_meas - self.ba
        w = gyro_meas - self.bg

        # rotation matrix from body to world
        q = quat_normalize(self.quat)
        R = quat_to_rotmat(q)

        # kinematic propagation
        self.x[0:3] = self.pos + self.vel * dt + 0.5 * (R @ a + GRAVITY) * dt * dt
        self.x[3:6] = self.vel + (R @ a + GRAVITY) * dt

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
        # vel_dot ~ R * (a) => partial wrt attitude ~ -R * [a]_x (attitude small-angle)
        ax = a
        # skew-symmetric of accel in body frame expressed in world: R @ ax cross
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

    def update_uwb_range(self, anchor_pos: np.ndarray, range_meas: float):
        """UWB range measurement to known anchor position in world frame.

        z (scalar) is the measured range to the anchor.
        """
        z = range_meas
        h = np.linalg.norm(self.pos - anchor_pos)
        if h < 1e-8:
            return
        # measurement Jacobian H (1x15) wrt error-state (pos, vel, att, ba, bg)
        H = np.zeros((1, 15))
        H[0, 0:3] = (self.pos - anchor_pos) / h

        S = H @ self.P @ H.T + self.R_uwb_range
        K = (self.P @ H.T) / S

        y = z - h
        dx = (K * y).flatten()

        self._inject_error_state(dx)

        # Joseph form
        self.P = (np.eye(15) - K @ H) @ self.P @ (np.eye(15) - K @ H).T + K * self.R_uwb_range * K.T

    def update_mag(self, mag_meas: np.ndarray, mag_ref: np.ndarray):
        """Magnetometer vector measurement (body frame) with reference vector in world frame.

        We predict mag in body frame: b_body = R.T @ mag_ref_world
        and compare to measured mag_meas.
        """
        q = quat_normalize(self.quat)
        R = quat_to_rotmat(q)
        b_pred = R.T @ mag_ref

        H = np.zeros((3, 15))
        # derivative wrt attitude: d(R.T v)/dtheta ~ -R.T [v]_x
        def skew(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        H[:, 6:9] = -R.T @ skew(mag_ref)

        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        y = mag_meas - b_pred
        dx = K @ y
        self._inject_error_state(dx.flatten())

        self.P = (np.eye(15) - K @ H) @ self.P

    def _inject_error_state(self, dx: np.ndarray):
        """Inject 15-vector error state into the full state x and renormalize quaternion.

        dx layout: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]
        """
        if dx.shape[0] != 15:
            raise ValueError("dx must be length 15")
        # pos
        self.x[0:3] = self.pos + dx[0:3]
        # vel
        self.x[3:6] = self.vel + dx[3:6]
        # attitude: apply small-angle
        dtheta = dx[6:9]
        dq = small_angle_quat(dtheta)
        q = self.quat
        q_new = quat_mul(dq, q)
        self.x[6:10] = quat_normalize(q_new)
        # accel bias
        self.x[10:13] = self.ba + dx[9:12]
        # gyro bias
        self.x[13:16] = self.bg + dx[12:15]