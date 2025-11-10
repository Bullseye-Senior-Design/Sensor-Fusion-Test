from structure.commands.Command import Command
import numpy as np
from Robot.MathUtil import MathUtil
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator


class AlignIMUCmd(Command):
    """Command that waits until the estimator's UWB-derived position has moved
    at least `distance_threshold` meters (planar XY). Once the threshold is
    reached it computes the motion heading and aligns the IMU yaw to that
    heading.

    Usage:
        cmd = AlignIMUCmd(distance_threshold=0.5, soft=True)
        cmd.schedule()

    Parameters:
        distance_threshold: meters of planar travel required to compute yaw
        soft: if True call estimator.update_imu_attitude(q) (soft fusion);
              if False directly set estimator state and tighten covariance.
    """

    def __init__(self, distance_threshold: float = 0.5, soft: bool = True):
        super().__init__()
        self.distance_threshold = float(distance_threshold)
        self.soft = bool(soft)

        self._est = KalmanStateEstimator()
        self._started = False
        self._start_pos = None
        self._applied = False

    def initalize(self):
        # record current position as start; if invalid, leave None and wait
        try:
            s = self._est.get_state()
            pos = np.asarray(s.pos, dtype=float)
            if np.all(np.isfinite(pos)):
                self._start_pos = pos.copy()
        except Exception:
            self._start_pos = None
        self._started = True

    def execute(self):
        if not self._started or self._applied:
            return

        try:
            s = self._est.get_state()
            pos = np.asarray(s.pos, dtype=float)
            if not np.all(np.isfinite(pos)):
                return

            if self._start_pos is None:
                # initialize start when a valid reading arrives
                self._start_pos = pos.copy()
                return

            # planar displacement (x,y)
            delta = pos[0:2] - self._start_pos[0:2]
            dist = np.linalg.norm(delta)
            if dist < self.distance_threshold:
                return

            # compute world heading from motion
            world_yaw = float(np.arctan2(delta[1], delta[0]))

            # get current IMU/estimator yaw
            q = self._est.quat
            imu_rpy = MathUtil.quat_to_euler(q)
            imu_yaw = float(imu_rpy[2])

            yaw_offset = world_yaw - imu_yaw

            # form quaternion to rotate by yaw_offset and apply
            q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, yaw_offset]))
            q_aligned = MathUtil.quat_mul(q_yaw, q)
            q_aligned = MathUtil.quat_normalize(q_aligned)

            if self.soft:
                # soft fusion through EKF attitude update
                try:
                    self._est.update_imu_attitude(q_aligned)
                except Exception:
                    # fallback to hard set below
                    self.soft = False

            if not self.soft:
                # hard set: overwrite quaternion and tighten covariance
                with self._est._lock:
                    self._est.x[6:10] = q_aligned
                    # tighten attitude cov (small-angle covariance)
                    self._est.P[6:9, 6:9] = np.eye(3) * 1e-6

            self._applied = True
            print(f"AlignIMUCmd: traveled {dist:.3f} m -> world_yaw={np.degrees(world_yaw):.2f} deg, imu_yaw={np.degrees(imu_yaw):.2f} deg, applied yaw_offset={np.degrees(yaw_offset):.2f} deg")

        except Exception as e:
            # don't crash the scheduler; just log
            print(f"AlignIMUCmd execute error: {e}")

    def end(self, interrupted):
        if interrupted:
            print("AlignIMUCmd: interrupted before alignment")

    def is_finished(self) -> bool:
        # finish after alignment applied
        return self._applied