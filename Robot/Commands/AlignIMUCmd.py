from structure.commands.Command import Command
import numpy as np
from Robot.MathUtil import MathUtil
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU

# TODO look into using PCA/SVD for a better offset estimate
class AlignIMUCmd(Command):
    """Command that waits until the estimator's UWB-derived position has moved
    at least `distance_threshold` meters (planar XY). Once the threshold is
    reached it computes the motion heading and aligns the IMU yaw to that
    heading.

    Parameters:
        distance_threshold: meters of planar travel required to compute yaw
    """

    def __init__(self, distance_threshold: float = 1):
        super().__init__()
        self.distance_threshold = float(distance_threshold)

        self._est = KalmanStateEstimator()
        self._uwb = UWB()
        self._imu = IMU()

    def initalize(self):
        self._started = False
        self._start_pos = None  # numpy array [x,y,z] in world (from UWB)
        self._applied = False
        
        # Try once to obtain an initial UWB reading; do not mark started
        # unless we have a valid UWB position.
        positions = self._uwb.get_positions()
        if positions:
            # average positions across available tags to reduce noise
            xs = [p.x for p in positions]
            ys = [p.y for p in positions]
            zs = [p.z for p in positions]
            self._start_pos = np.array([float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))], dtype=float)
            self._started = True

    def execute(self):
        # Ensure we have a valid UWB start position. If not started yet,
        # try to obtain one from the UWB subsystem.
        if not self._started:
            try:
                positions = self._uwb.get_positions()
            except Exception:
                positions = []
            if not positions:
                # no UWB reading yet; wait
                return
            # average across tags
            xs = [p.x for p in positions]
            ys = [p.y for p in positions]
            zs = [p.z for p in positions]
            self._start_pos = np.array([float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))], dtype=float)
            self._started = True
            # continue to next execute loop (we have start pos now)
            return

        # Get current UWB position(s)
        positions = self._uwb.get_positions()

        if not positions:
            # no valid current reading; wait
            return

        # average across available tags for current position
        xs = [p.x for p in positions]
        ys = [p.y for p in positions]
        zs = [p.z for p in positions]
        cur_pos = np.array([float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))], dtype=float)

        # planar displacement (x,y)
        delta = cur_pos[0:2] - self._start_pos[0:2] # type: ignore
        dist = np.linalg.norm(delta)
        if dist < self.distance_threshold:
            return

        # compute world heading from motion (radians)
        world_yaw = float(np.arctan2(delta[1], delta[0]))
        print(f"AlignIMUCmd: delta={delta}, world_yaw={np.degrees(world_yaw):.2f} deg")


        # get current IMU/estimator yaw (from estimator quaternion)
        q_est = self._est.quat
        imu_rpy = MathUtil.quat_to_euler(q_est)
        imu_yaw = float(imu_rpy[2])

        yaw_offset = world_yaw - imu_yaw  # radians

        # Set yaw offset in IMU (degrees)
        self._imu.set_yaw_offset(np.degrees(yaw_offset))

        self._applied = True
        print(f"AlignIMUCmd: traveled {dist:.3f} m -> world_yaw={np.degrees(world_yaw):.2f} deg, imu_yaw={np.degrees(imu_yaw):.2f} deg, applied yaw_offset={np.degrees(yaw_offset):.2f} deg")

    def end(self, interrupted):
        if interrupted:
            print("AlignIMUCmd: interrupted before alignment")

    def is_finished(self) -> bool:
        # finish after alignment applied
        return self._applied