from structure.commands.Command import Command


import csv
import os
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

# subsystems
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.UWBTag import Position
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator

class LogDataCmd(Command):
    def __init__(self):
        super().__init__()
        
    def initalize(self):
        self.begin_timestamp = time.time()
        print("LogDataCmd initialized, logging started.")
    
    def execute(self):
        """Sample UWB positions, estimator state, and IMU orientation once and append to CSVs.

        This method is designed to be called repeatedly by the command scheduler
        (it performs one sample per call). It will gracefully skip subsystems
        that are not available/initialized.
        """
        ts = time.time()

        # Ensure log directory
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # helper to create filenames stamped with the command begin timestamp
        def _make_log_filename(base: str) -> str:
            # use the begin timestamp if available so all files share same prefix
            ts0 = getattr(self, 'begin_timestamp', None)
            if ts0 is None:
                ts0 = time.time()
            ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts0))
            return str(log_dir / f"{base}_{ts_str}.csv")

        # 1) UWB positions
        try:
            uwb = UWB()
            positions = []
            if hasattr(uwb, 'get_positions'):
                positions = uwb.get_positions() or []
            # take up to two positions (older code expected two)
            p1 = positions[0] if len(positions) > 0 else None
            p2 = positions[1] if len(positions) > 1 else None
            uwb_file = _make_log_filename('uwb_positions')
            self.save_uwb_pos_to_csv(p1, p2, uwb_file, timestamp=ts)
        except Exception as e:
            print(f"UWB read error: {e}")

        # 2) State estimator
        try:
            kf = KalmanStateEstimator()
            state = kf.get_state()
            # estimate euler from estimator for convenience
            euler = kf.euler  # numpy array [roll, pitch, yaw] in radians
            yaw = float(euler[2]) * 180.0 / 3.141592653589793
            pitch = float(euler[1]) * 180.0 / 3.141592653589793
            roll = float(euler[0]) * 180.0 / 3.141592653589793
            state_file = _make_log_filename('state_estimator')
            self.save_state_to_csv(state, yaw, pitch, roll, state_file, timestamp=ts)
        except Exception as e:
            print(f"State estimator read error: {e}")

        # 3) IMU orientation
        try:
            imu = IMU()
            # IMU may be running in its own thread; get_euler returns (heading, roll, pitch)
            heading, roll, pitch = imu.get_euler()
            orient = SimpleNamespace(timestamp=ts, yaw=heading, pitch=pitch, roll=roll)
            imu_file = _make_log_filename('imu_orientation')
            self.save_orientation_to_csv(orient, imu_file)
        except Exception as e:
            print(f"IMU read error: {e}")
    
    def end(self, interrupted):
        pass
    
    def is_finished(self):
        return False
    
    def save_uwb_pos_to_csv(self, position1: Optional[Position], position2: Optional[Position], filename: str, timestamp: Optional[float] = None) -> bool:
        """
        Save a position reading to a CSV file
        
        Args:
            position: Position object to save
            filename: CSV filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Backwards-compatible wrapper: accept None values and optional timestamp
        def _safe_get(p: Optional[Position], attr, default=''):
            return getattr(p, attr) if (p is not None and hasattr(p, attr)) else default

        try:
            # allow filename as Path
            filename = str(filename)
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(filename)

            # Open file in append mode
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'x1', 'y1', 'z1', 'quality1', 'x2', 'y2', 'z2', 'quality2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")

                ts = timestamp if timestamp is not None else time.time()

                # Write position data (allow missing positions)
                writer.writerow({
                    'timestamp': ts,
                    'x1': _safe_get(position1, 'x', ''),
                    'y1': _safe_get(position1, 'y', ''),
                    'z1': _safe_get(position1, 'z', ''),
                    'quality1': _safe_get(position1, 'quality', ''),
                    'x2': _safe_get(position2, 'x', ''),
                    'y2': _safe_get(position2, 'y', ''),
                    'z2': _safe_get(position2, 'z', ''),
                    'quality2': _safe_get(position2, 'quality', ''),
                })

            return True
        except Exception as e:
            print(f"Error saving position to CSV: {e}")
            return False
    
    def save_orientation_to_csv(self, orientation, filename: str) -> bool:
        """
        Save a position reading to a CSV file
        
        Args:
            position: Position object to save
            filename: CSV filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = str(filename)
            file_exists = os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'yaw', 'pitch', 'roll']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")

                ts = getattr(orientation, 'timestamp', time.time())
                writer.writerow({
                    'timestamp': ts,
                    'yaw': getattr(orientation, 'yaw', ''),
                    'pitch': getattr(orientation, 'pitch', ''),
                    'roll': getattr(orientation, 'roll', ''),
                })

            return True
        except Exception as e:
            print(f"Error saving orientation to CSV: {e}")
            return False

    def save_state_to_csv(self, state, yaw: float, pitch: float, roll: float, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save estimator state (pos, vel, euler) to CSV."""
        try:
            filename = str(filename)
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")

                ts = timestamp if timestamp is not None else time.time()
                px, py, pz = state.pos
                vx, vy, vz = state.vel
                writer.writerow({
                    'timestamp': ts,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                })
            return True
        except Exception as e:
            print(f"Error saving state to CSV: {e}")
            return False