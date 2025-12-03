from structure.commands.Command import Command
import time
import math
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.sensors.UWB import UWB
import logging

logger = logging.getLogger(__name__)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AlignIMUToWorldCmd(Command):
    """Command to estimate and apply a yaw offset between IMU and UWB.

    This implements a scalar bias estimator (integrator-like) that updates
    the IMU yaw offset via `IMU().set_yaw_offset(...)` so the IMU and UWB
    headings align. The estimator runs until `duration` elapses or the
    residual is stable for several samples.
    """
    def __init__(self, tau: float = 5.0, duration: float = 30.0, tol_rad: float = 0.01, min_samples: int = 20):
        super().__init__()
        # time constant (seconds) for bias adaptation
        self.tau = float(tau)
        # maximum command run time (seconds)
        self.duration = float(duration)
        # convergence tolerance on yaw residual (radians)
        self.tol = float(tol_rad)
        # minimum samples before allowing early finish
        self.min_samples = int(min_samples)

        # runtime state
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._samples = 0
        self._stable_count = 0
        self._bias = 0.0  # radians

    def initialize(self):
        self._start_time = time.time()
        self._last_time = self._start_time
        self._samples = 0
        self._stable_count = 0
        # Don't reset the IMU yaw offset here; we'll adapt from current
        logger.info(f"AlignIMUToWorldCmd: starting yaw-bias estimation (tau={self.tau}s, duration={self.duration}s)")

    def execute(self):
        now = time.time()
        if self._last_time is None:
            self._last_time = now
        dt = max(1e-6, now - self._last_time)

        # Get instantaneous UWB yaw (radians)
        uwb = UWB()
        uwb_yaw = uwb.get_angle()
        if uwb_yaw is None:
            #  logger.warning("AlignIMUToWorldCmd: insufficient UWB tags to compute heading; skipping this cycle")
            self._start_time = time.time()  # reset start time to avoid premature timeout
            return
        
        # Get IMU yaw in radians (IMU.get_euler returns degrees: (yaw, roll, pitch))
        imu = IMU()
        imu_euler = imu.get_euler()
        imu_yaw_deg = float(imu_euler[0])
        imu_yaw_rad = math.radians(imu_yaw_deg)

        logger.info(f"AlignIMUToWorldCmd: UWB yaw = {math.degrees(uwb_yaw):.3f} deg")
        logger.info(f"AlignIMUToWorldCmd: IMU yaw = {imu_yaw_deg:.3f} deg (with offset {math.degrees(self._bias):.3f} deg)")

        # residual between measured UWB yaw and corrected IMU yaw
        residual = _wrap_angle(uwb_yaw - (imu_yaw_rad + self._bias))

        # compute alpha from tau
        alpha = dt / (self.tau + dt)

        # integrator-like update
        self._bias += alpha * residual

        # apply bias to IMU (degrees)
        logger.info(f"AlignIMUToWorldCmd: applying yaw offset {math.degrees(self._bias):.3f} deg")
        imu.set_yaw_offset(math.degrees(self._bias))


        # bookkeeping
        self._samples += 1
        if abs(residual) < self.tol:
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._last_time = now

    def end(self, interrupted):
        if interrupted:
            logger.info("AlignIMUToWorldCmd interrupted; leaving current IMU yaw offset in place")
        else:
            logger.info(f"AlignIMUToWorldCmd completed: applied yaw offset {math.degrees(self._bias):.3f} deg")

    def is_finished(self) -> bool:
        # shouldn't happen, but guard against None
        if self._start_time is None:
            return True
        
        elapsed = time.time() - self._start_time
        if elapsed >= self.duration:
            return True
        if self._samples >= self.min_samples and self._stable_count >= 10:
            return True
        return False