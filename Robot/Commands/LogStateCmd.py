from structure.commands.Command import Command


import csv
import os
import time
import json
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
        # record when logging begins and create a timestamped folder to hold all logs
        self.begin_timestamp = time.time()
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.begin_timestamp))

        # create a dedicated folder under ./logs/<begin_timestamp>/
        base_log_dir = Path.cwd() / 'logs'
        self.log_dir = base_log_dir / ts_str
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"LogDataCmd initialized, logging started. Logs will be stored in: {self.log_dir}")
    
    def execute(self):
        """Sample UWB positions, estimator state, and IMU orientation once and append to CSVs.

        This method is designed to be called repeatedly by the command scheduler
        (it performs one sample per call). It will gracefully skip subsystems
        that are not available/initialized.
        """
        ts = time.time()

        # fallback: if execute is called before initalize, ensure a log dir exists
        if not hasattr(self, 'log_dir'):
            ts0 = getattr(self, 'begin_timestamp', time.time())
            ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts0))
            self.log_dir = Path.cwd() / 'logs' / ts_str
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # helper to create filenames inside the command's timestamped folder
        def _make_log_filename(base: str, ext: str = 'csv') -> str:
            return str(self.log_dir / f"{base}.{ext}")

        # 1) UWB positions
        try:
            uwb = UWB()
            positions = []
            if hasattr(uwb, 'get_positions'):
                positions = uwb.get_positions() or []
            # take up to two positions (older code expected two)
            p1 = positions[0] if len(positions) > 0 else None
            p2 = positions[1] if len(positions) > 1 else None
            uwb_file = _make_log_filename('uwb_positions', 'csv')
            self.save_uwb_pos_to_csv(p1, p2, uwb_file, timestamp=ts)

            # Also record anchor information (text file) for debugging / reference
            try:
                if hasattr(uwb, 'get_latest_anchor_info'):
                    anchors = uwb.get_latest_anchor_info()
                    anchor_file = _make_log_filename('uwb_anchors', 'csv')
                    self.save_uwb_anchors_to_csv(anchors, anchor_file, timestamp=ts)
            except Exception as e:
                print(f"UWB anchor save error: {e}")
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
            # Also log covariance matrix (EKF P)
            try:
                cov_file = _make_log_filename('ekf_covariance', 'csv')
                if hasattr(kf, 'P'):
                    self.save_covariance_to_csv(kf.P, cov_file, timestamp=ts)
            except Exception as e:
                print(f"Covariance save error: {e}")
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

    def save_uwb_anchors_to_csv(self, anchors_info, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save UWB anchor info into a CSV file.

        anchors_info: expected to be the list returned by UWB.get_latest_anchor_info(),
                      i.e., a list of tuples (port, anchors) where anchors is a list of dicts
                      (each dict has keys 'name','id','position'=(x,y,z),'range').
        The CSV columns will be: timestamp, port, name, id, x, y, z, range
        """
        try:
            filename = str(filename)
            file_exists = os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'port', 'name', 'id', 'x', 'y', 'z', 'range']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")

                ts = timestamp if timestamp is not None else time.time()

                # anchors_info is list of (port, anchors) tuples
                for port, anchors in anchors_info or []:
                    if not anchors:
                        # write a row indicating no anchors for this port
                        writer.writerow({'timestamp': ts, 'port': port, 'name': '', 'id': '', 'x': '', 'y': '', 'z': '', 'range': ''})
                        continue
                    for anchor in anchors:
                        name = anchor.get('name', '')
                        aid = anchor.get('id', '')
                        pos = anchor.get('position', (None, None, None))
                        rng = anchor.get('range', '')
                        x = pos[0] if pos and len(pos) > 0 else ''
                        y = pos[1] if pos and len(pos) > 1 else ''
                        z = pos[2] if pos and len(pos) > 2 else ''
                        writer.writerow({'timestamp': ts, 'port': port, 'name': name, 'id': aid, 'x': x, 'y': y, 'z': z, 'range': rng})

            return True
        except Exception as e:
            print(f"Error saving UWB anchors to CSV: {e}")
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

    def save_covariance_to_csv(self, P, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save a square covariance matrix P as a single row in CSV.

        Columns: timestamp, p00, p01, ..., p0n, p10, p11, ..., pnn
        """
        try:
            filename = str(filename)
            file_exists = os.path.exists(filename)

            # convert to nested list if numpy array
            try:
                rows = P.tolist()
            except Exception:
                # assume P is already list-like
                rows = [list(r) for r in P]

            n = len(rows)
            # validate square
            if any(len(r) != n for r in rows):
                print(f"Covariance matrix is not square: {filename}")
                return False

            # build fieldnames
            fieldnames = ['timestamp'] + [f'p{i}{j}' for i in range(n) for j in range(n)]

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")

                ts = timestamp if timestamp is not None else time.time()
                flat = {}
                flat['timestamp'] = ts
                for i in range(n):
                    for j in range(n):
                        flat[f'p{i}{j}'] = rows[i][j]

                writer.writerow(flat)

            return True
        except Exception as e:
            print(f"Error saving covariance to CSV: {e}")
            return False