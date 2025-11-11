from structure.commands.Command import Command
import numpy as np
from Robot.MathUtil import MathUtil
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU

class AlignIMUCmd(Command):
    """Command that waits until the estimator's UWB-derived position has moved
    at least `distance_threshold` meters (planar XY). Once the threshold is
    reached it computes the motion heading and aligns the IMU yaw to the x+ axis
    of the motion heading.

    Parameters:
        distance_threshold: meters of planar travel required to compute yaw
    """

    def __init__(self, distance_threshold: float = 1):
        super().__init__()
        self.distance_threshold = float(distance_threshold)

        self._uwb = UWB()
        self._imu = IMU()
        # store recent UWB positions (list of [x,y] numpy arrays) for PCA
        self._positions_history = []

    def initailize(self):
        self._started = False
        self._start_pos = None  # numpy array [x,y,z] in world (from UWB)
        self._applied = False
        self._positions_history = []
        
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

        pos2 = cur_pos[0:2]  # type: ignore
        # append to history (keep as Python floats / small list)
        self._positions_history.append(np.array([float(pos2[0]), float(pos2[1])], dtype=float))

        # distance from start (planar)
        delta = pos2 - self._start_pos[0:2]  # type: ignore
        dist = np.linalg.norm(delta)
        if dist < self.distance_threshold:
            return

        # We have travelled far enough; estimate the dominant motion direction
        # using PCA / SVD over the collected 2D positions.
        if len(self._positions_history) >= 1:
            # stack history into an (N,2) array
            positions_arr = np.vstack(self._positions_history)
        else:
            positions_arr = np.array([pos2])

        # If there are too few samples, fall back to start->current delta
        if positions_arr.shape[0] < 2:
            world_yaw = float(np.arctan2(delta[1], delta[0]))
            explained = 1.0
        else:
            world_yaw, explained = self._estimate_motion_heading(positions_arr)

        print(f"AlignIMUCmd: traveled {dist:.3f} m, movement_heading={np.degrees(world_yaw):.2f} deg, explained_variance={explained:.2f}")

        # If the positions are not sufficiently linear, warn and fall back
        if explained < 0.5:
            print("AlignIMUCmd: PCA indicates poor linear motion (explained variance < 0.5); alignment skipped")
            # clear history so next motion attempt starts fresh
            self._positions_history = []
            return

        # get current IMU/estimator yaw (from estimator quaternion)
        imu_euler = self._imu.get_euler()
        imu_yaw = float(imu_euler[0])

        imu_yaw_deg = imu_yaw
        imu_yaw_rad = np.radians(imu_yaw_deg)

        # compute yaw offset to align IMU heading to the motion heading
        yaw_offset = float(world_yaw) - imu_yaw_rad  # radians

        # normalize to [-pi, pi]
        yaw_offset = (yaw_offset + np.pi) % (2 * np.pi) - np.pi

        # Set yaw offset in IMU (degrees)
        self._imu.set_yaw_offset(np.degrees(yaw_offset))

        self._applied = True
        print(f"AlignIMUCmd: applied yaw_offset={np.degrees(yaw_offset):.2f} deg (imu_yaw={imu_yaw_deg:.2f} deg, motion={np.degrees(world_yaw):.2f} deg)")

    def end(self, interrupted):
        if interrupted:
            print("AlignIMUCmd: interrupted before alignment")

    def is_finished(self) -> bool:
        # finish after alignment applied
        return self._applied

    def _estimate_motion_heading(self, positions_2d: np.ndarray):
        """Estimate dominant motion direction (radians) from Nx2 positions using SVD.

        Returns (angle_radians, explained_variance_ratio).
        explained_variance_ratio = s0 / (s0 + s1) where s are singular values.
        """
        pts = np.asarray(positions_2d, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("positions_2d must be an Nx2 array")
        # Center points
        mean = pts.mean(axis=0)
        centered = pts - mean
        # SVD on centered data (rows = samples)
        try:
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            # fallback to simple start->end
            delta = pts[-1] - pts[0]
            angle = float(np.arctan2(delta[1], delta[0]))
            return angle, 0.0

        # principal direction is first right-singular vector
        principal = vh[0]
        angle = float(np.arctan2(principal[1], principal[0]))
        # explained variance (proportion) from singular values
        if s.size >= 2:
            explained = float(s[0] / (s[0] + s[1]))
        else:
            explained = 1.0
        return angle, explained