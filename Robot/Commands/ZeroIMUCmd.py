from Robot.Commands import FollowPathCmd
from structure.commands.Command import Command
from Robot.subsystems.sensors.IMU import IMU
import numpy as np
from Robot.Commands.FollowPathCmd import FollowPathCmd
from Robot.subsystems.MotorControl import MotorControl
from Robot.subsystems.PathFollowing import PathFollowing


class ZeroIMUCmd(Command):
    """Command that sets the IMU yaw offset so the current heading becomes zero.

    Behavior:
    - Collect up to `sample_count` heading samples (degrees) via
      `IMU.get_euler()` and compute a circular-safe median by unwrapping
      radians before taking the median.
    - Apply yaw offset = -median_heading (degrees) using
      `IMU.set_yaw_offset()` and finish.
    """

    def __init__(self,
                 motor_control: MotorControl,
                 path_following: PathFollowing,
                 sample_count: int = 10):
        super().__init__()
        self._imu = IMU()
        self.motor_control = motor_control
        self.path_following = path_following
        self._applied = False
        self.sample_count = int(sample_count)
        self._samples = []

    def initialize(self):
        self._applied = False
        self._samples = []

    def execute(self):
        if self._applied:
            return

        try:
            euler = self._imu.get_euler()
        except Exception as e:
            print(f"ZeroIMUCmd: IMU.get_euler() exception: {e}")
            return

        if not euler or len(euler) < 1:
            return

        try:
            heading_deg = float(euler[0])
        except Exception:
            return

        # store sample
        self._samples.append(heading_deg)
        # keep at most sample_count (older values are fine but we only need N)
        if len(self._samples) < self.sample_count:
            # not enough samples yet
            return

        # compute circular-safe median:
        # - convert to radians
        # - unwrap to remove wrap discontinuities
        # - take median
        # - convert back to degrees and wrap to [-180, 180]
        arr = np.asarray(self._samples, dtype=float)
        radians = np.radians(arr)
        unwrapped = np.unwrap(radians)
        median_rad = float(np.median(unwrapped))
        median_deg = np.degrees(median_rad)
        # normalize to [-180,180]
        median_deg = ((median_deg + 180.0) % 360.0) - 180.0

        yaw_offset_deg = -median_deg
        self._imu.set_yaw_offset(yaw_offset_deg)
        self._applied = True
        print(f"ZeroIMUCmd: applied yaw offset {yaw_offset_deg:.2f} deg to zero median heading (median was {median_deg:.2f} deg)")

    def end(self, interrupted):
        if interrupted and not self._applied:
            print("ZeroIMUCmd: interrupted before applying yaw offset")
        FollowPathCmd(self.motor_control, self.path_following).schedule()

    def is_finished(self):
        return self._applied