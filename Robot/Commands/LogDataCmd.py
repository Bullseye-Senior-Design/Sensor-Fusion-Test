from structure.commands.Command import Command


import csv
import os
import time
import json
import numpy as np
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
        
    def initialize(self):
        # record when logging begins and create a timestamped folder to hold all logs
        self.begin_timestamp = time.time()
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.begin_timestamp))

        # create a dedicated folder under ./logs/<begin_timestamp>/
        base_log_dir = Path.cwd() / 'logs'
        self.log_dir = base_log_dir / ts_str
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"LogDataCmd initialized, logging started. Logs will be stored in: {self.log_dir}")
        # Open persistent file handles and CSV writers to avoid reopening files every sample
        try:
            # file paths
            self.uwb_file_path = str(self.log_dir / 'uwb_positions.csv')
            self.anchors_file_path = str(self.log_dir / 'uwb_anchors.csv')
            self.state_file_path = str(self.log_dir / 'state_estimator.csv')
            self.imu_file_path = str(self.log_dir / 'imu_orientation.csv')
            self.cov_file_path = str(self.log_dir / 'ekf_covariance.txt')

            # open UWB positions CSV
            self._uwb_fh = open(self.uwb_file_path, 'a', newline='')
            uwb_new = os.path.getsize(self.uwb_file_path) == 0
            uwb_fieldnames = ['timestamp', 'x1', 'y1', 'z1', 'quality1', 'x2', 'y2', 'z2', 'quality2']
            self._uwb_writer = csv.DictWriter(self._uwb_fh, fieldnames=uwb_fieldnames)
            if uwb_new:
                self._uwb_writer.writeheader()

            # open anchors CSV
            self._anchors_fh = open(self.anchors_file_path, 'a', newline='')
            anchors_new = os.path.getsize(self.anchors_file_path) == 0
            anchors_fieldnames = ['timestamp', 'port', 'name', 'id', 'x', 'y', 'z', 'range']
            self._anchors_writer = csv.DictWriter(self._anchors_fh, fieldnames=anchors_fieldnames)
            if anchors_new:
                self._anchors_writer.writeheader()

            # open state CSV
            self._state_fh = open(self.state_file_path, 'a', newline='')
            state_new = os.path.getsize(self.state_file_path) == 0
            state_fieldnames = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll']
            self._state_writer = csv.DictWriter(self._state_fh, fieldnames=state_fieldnames)
            if state_new:
                self._state_writer.writeheader()

            # open imu CSV
            self._imu_fh = open(self.imu_file_path, 'a', newline='')
            imu_new = os.path.getsize(self.imu_file_path) == 0
            imu_fieldnames = ['timestamp', 'yaw', 'pitch', 'roll', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
            self._imu_writer = csv.DictWriter(self._imu_fh, fieldnames=imu_fieldnames)
            if imu_new:
                self._imu_writer.writeheader()

            # open covariance text file handle for append
            self._cov_fh = open(self.cov_file_path, 'a')
            cov_new = os.path.getsize(self.cov_file_path) == 0
            if cov_new:
                # write a small header block
                self._cov_fh.write(f"# ekf covariance log started at {self.begin_timestamp}\n")
                self._cov_fh.flush()
        except Exception as e:
            print(f"LogDataCmd: failed to open persistent log files: {e}")
    
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
                cov_file = _make_log_filename('ekf_covariance', 'txt')
                if hasattr(kf, 'P'):
                    self.save_covariance_to_txt(kf.P, cov_file, timestamp=ts)
            except Exception as e:
                print(f"Covariance save error: {e}")
        except Exception as e:
            print(f"State estimator read error: {e}")

        # 3) IMU orientation
        try:
            imu = IMU()
            # IMU may be running in its own thread; get_euler returns (heading, roll, pitch)
            heading, roll, pitch = imu.get_euler()
            # get raw sensor measurements (accel, gyro, mag)
            try:
                accel = imu.get_accel()
            except Exception:
                accel = None
            try:
                gyro = imu.get_gyro()
            except Exception:
                gyro = None
            try:
                mag = imu.get_mag()
            except Exception:
                mag = None

            orient = SimpleNamespace(timestamp=ts, yaw=heading, pitch=pitch, roll=roll, accel=accel, gyro=gyro, mag=mag)
            imu_file = _make_log_filename('imu_orientation')
            # save orientation and raw sensor values
            self.save_orientation_to_csv(orient, imu_file)
        except Exception as e:
            print(f"IMU read error: {e}")
    
    def end(self, interrupted):
        # Close any persistent file handles opened in initialize
        if hasattr(self, '_uwb_fh') and not self._uwb_fh.closed:
            self._uwb_fh.close()
        if hasattr(self, '_anchors_fh') and not self._anchors_fh.closed:
            self._anchors_fh.close()
        if hasattr(self, '_state_fh') and not self._state_fh.closed:
            self._state_fh.close()
        if hasattr(self, '_imu_fh') and not self._imu_fh.closed:
            self._imu_fh.close()
        if hasattr(self, '_cov_fh') and not self._cov_fh.closed:
            self._cov_fh.close()
    
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
            ts = timestamp if timestamp is not None else time.time()

            # If persistent writer is available and filename matches, use it
            if hasattr(self, '_uwb_writer') and hasattr(self, 'uwb_file_path') and str(filename) == str(self.uwb_file_path):
                self._uwb_writer.writerow({
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
                try:
                    self._uwb_fh.flush()
                except Exception:
                    pass
                return True

            # Fallback: open file each time (existing behavior)
            filename = str(filename)
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'x1', 'y1', 'z1', 'quality1', 'x2', 'y2', 'z2', 'quality2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
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
            ts = getattr(orientation, 'timestamp', time.time())

            def _triplet(val):
                if val is None:
                    return ('', '', '')
                try:
                    a, b, c = val
                    return (a if a is not None else '', b if b is not None else '', c if c is not None else '')
                except Exception:
                    return ('', '', '')

            ax, ay, az = _triplet(getattr(orientation, 'accel', None))
            gx, gy, gz = _triplet(getattr(orientation, 'gyro', None))
            mx, my, mz = _triplet(getattr(orientation, 'mag', None))

            # Use persistent writer if available and filename matches
            if hasattr(self, '_imu_writer') and hasattr(self, 'imu_file_path') and filename == str(self.imu_file_path):
                self._imu_writer.writerow({
                    'timestamp': ts,
                    'yaw': getattr(orientation, 'yaw', ''),
                    'pitch': getattr(orientation, 'pitch', ''),
                    'roll': getattr(orientation, 'roll', ''),
                    'ax': ax,
                    'ay': ay,
                    'az': az,
                    'gx': gx,
                    'gy': gy,
                    'gz': gz,
                    'mx': mx,
                    'my': my,
                    'mz': mz,
                })
                try:
                    self._imu_fh.flush()
                except Exception:
                    pass
                return True

            # Fallback: open file each time
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'yaw', 'pitch', 'roll',
                              'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'timestamp': ts,
                    'yaw': getattr(orientation, 'yaw', ''),
                    'pitch': getattr(orientation, 'pitch', ''),
                    'roll': getattr(orientation, 'roll', ''),
                    'ax': ax,
                    'ay': ay,
                    'az': az,
                    'gx': gx,
                    'gy': gy,
                    'gz': gz,
                    'mx': mx,
                    'my': my,
                    'mz': mz,
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
            ts = timestamp if timestamp is not None else time.time()

            # Use persistent anchors writer if available
            if hasattr(self, '_anchors_writer') and hasattr(self, 'anchors_file_path') and filename == str(self.anchors_file_path):
                for port, anchors in anchors_info or []:
                    if not anchors:
                        self._anchors_writer.writerow({'timestamp': ts, 'port': port, 'name': '', 'id': '', 'x': '', 'y': '', 'z': '', 'range': ''})
                        continue
                    for anchor in anchors:
                        name = anchor.get('name', '')
                        aid = anchor.get('id', '')
                        pos = anchor.get('position', (None, None, None))
                        rng = anchor.get('range', '')
                        x = pos[0] if pos and len(pos) > 0 else ''
                        y = pos[1] if pos and len(pos) > 1 else ''
                        z = pos[2] if pos and len(pos) > 2 else ''
                        self._anchors_writer.writerow({'timestamp': ts, 'port': port, 'name': name, 'id': aid, 'x': x, 'y': y, 'z': z, 'range': rng})
                try:
                    self._anchors_fh.flush()
                except Exception:
                    pass
                return True

            # Fallback: open file each time
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'port', 'name', 'id', 'x', 'y', 'z', 'range']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                ts = timestamp if timestamp is not None else time.time()
                for port, anchors in anchors_info or []:
                    if not anchors:
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
        """Save estimator state (pos, vel, euler) and biases to CSV.

        ba: optional 3-tuple (bax, bay, baz)
        bg: optional 3-tuple (bgx, bgy, bgz)
        """
        try:
            ts = timestamp if timestamp is not None else time.time()
            px, py, pz = state.pos
            vx, vy, vz = state.vel

            # Use persistent writer if available
            if hasattr(self, '_state_writer') and hasattr(self, 'state_file_path') and str(filename) == str(self.state_file_path):
                self._state_writer.writerow({
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
                try:
                    self._state_fh.flush()
                except Exception:
                    pass
                return True

            filename = str(filename)
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
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

    def save_covariance_to_txt(self, P, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save covariance matrix in a readable text file.

        The file will contain a timestamp header followed by the matrix rows.
        Multiple calls append additional blocks (so the file contains history).
        """
        try:
            filename = str(filename)
            file_exists = os.path.exists(filename)

            # ensure numpy array
            P_arr = np.asarray(P)
            if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
                print(f"Covariance must be a square 2D array, got shape {P_arr.shape}")
                return False

            ts = timestamp if timestamp is not None else time.time()

            with open(filename, 'a') as f:
                # write a small block header for readability
                f.write(f"# timestamp: {ts}\n")
                # write matrix rows with nice formatting
                # use scientific notation with reasonable precision
                for row in P_arr:
                    f.write('  '.join(f"{val: .6e}" for val in row) + "\n")
                f.write("\n")

            if not file_exists:
                print(f"Created new covariance TXT file: {filename}")

            return True
        except Exception as e:
            print(f"Error saving covariance to TXT: {e}")
            return False