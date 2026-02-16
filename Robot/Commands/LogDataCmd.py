from structure.commands.Command import Command

import csv
import os
import time
import json
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from typing import Optional
import shutil

# subsystems
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.UWBTag import Position
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.sensors.BackWheelEncoder import BackWheelEncoder
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.PathFollowing import PathFollowing

# CSV utilities
from Robot.Commands.log_data.csvlib import CSVFileManager, write_csv_or_fallback

class LogDataCmd(Command):
    # CSV field names
    UWB_FIELDNAMES = ['timestamp', 'x1', 'y1', 'z1', 'quality1', 'x2', 'y2', 'z2', 'quality2']
    ANCHORS_FIELDNAMES = ['timestamp', 'port', 'name', 'id', 'x', 'y', 'z', 'range']
    STATE_FIELDNAMES = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll']
    IMU_FIELDNAMES = ['timestamp', 'yaw', 'pitch', 'roll', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
    ENCODER_FIELDNAMES = ['timestamp', 'count', 'velocity']
    PATH_FOLLOWING_FIELDNAMES = ['timestamp', 'motor_speed_mps', 'steering_angle_rad', 'steering_angle_deg']
    
    def __init__(self, path_following: PathFollowing):
        super().__init__()
        self.path_following = path_following
        self.csv_manager = CSVFileManager()
    
    def _delete_old_folders(self, base_dir, max_age_days=7):
        """Delete folders in base_dir that are older than max_age_days.
        
        Assumes folder names follow YYYYMMDD_HHMMSS format.
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for folder in base_path.iterdir():
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            # Parse the folder name as a timestamp
            # Expected format: YYYYMMDD_HHMMSS
            if len(folder_name) >= 15 and folder_name[8] == '_':
                folder_time = time.mktime(time.strptime(folder_name[:15], '%Y%m%d_%H%M%S'))
                age_seconds = current_time - folder_time
                
                if age_seconds > max_age_seconds:
                    print(f"Deleting old folder: {folder}")
                    shutil.rmtree(folder)
        
    def initialize(self):
        # Clean up old log folders (older than 1 week)
        self._delete_old_folders(Path.cwd() / 'logs', max_age_days=7)
        
        # record when logging begins and create a timestamped folder to hold all logs
        self.begin_timestamp = time.time()
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.begin_timestamp))

        # create a dedicated folder under ./logs/<begin_timestamp>/
        base_log_dir = Path.cwd() / 'logs'
        self.log_dir = base_log_dir / ts_str
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"LogDataCmd initialized, logging started. Logs will be stored in: {self.log_dir}")
        # Setup CSV files using manager
        # file paths
        self.uwb_file_path = str(self.log_dir / 'uwb_positions.csv')
        self.anchors_file_path = str(self.log_dir / 'uwb_anchors.csv')
        self.state_file_path = str(self.log_dir / 'state_estimator.csv')
        self.imu_file_path = str(self.log_dir / 'imu_orientation.csv')
        self.cov_file_path = str(self.log_dir / 'ekf_covariance.txt')
        self.encoder_file_path = str(self.log_dir / 'encoder_data.csv')
        self.path_following_file_path = str(self.log_dir / 'path_following.csv')

        # Setup CSV files with manager
        self.csv_manager.setup_file(self.uwb_file_path, self.UWB_FIELDNAMES)
        self.csv_manager.setup_file(self.anchors_file_path, self.ANCHORS_FIELDNAMES)
        self.csv_manager.setup_file(self.state_file_path, self.STATE_FIELDNAMES)
        self.csv_manager.setup_file(self.imu_file_path, self.IMU_FIELDNAMES)
        self.csv_manager.setup_file(self.encoder_file_path, self.ENCODER_FIELDNAMES)
        self.csv_manager.setup_file(self.path_following_file_path, self.PATH_FOLLOWING_FIELDNAMES)
        
        # Setup covariance text file separately (not CSV)
        self._cov_fh = open(self.cov_file_path, 'a')
        cov_new = os.path.getsize(self.cov_file_path) == 0
        if cov_new:
            self._cov_fh.write(f"# ekf covariance log started at {self.begin_timestamp}\n")
            self._cov_fh.flush()
    
    def execute(self):
        """Sample UWB positions, estimator state, and IMU orientation once and append to CSVs.

        This method is designed to be called repeatedly by the command scheduler
        (it performs one sample per call). It will gracefully skip subsystems
        that are not available/initialized.
        """
        ts = time.time()

        # 1) UWB positions
        uwb = UWB()
        positions = uwb.get_positions() or []
            
        # take up to two positions (older code expected two)
        p1 = positions[0] if len(positions) > 0 else None
        p2 = positions[1] if len(positions) > 1 else None
        self.save_uwb_pos_to_csv(p1, p2, self.uwb_file_path, timestamp=ts)

        # Also record anchor information (text file) for debugging / reference
        anchors = uwb.get_latest_anchor_info()
        self.save_uwb_anchors_to_csv(anchors, self.anchors_file_path, timestamp=ts)

        # 2) State estimator (only log if initialized)
        kf = KalmanStateEstimator()
        # Only log state if the filter has been initialized with first UWB measurement
        if kf.is_initialized:
            state = kf.get_state()
            # estimate euler from estimator for convenience
            euler = kf.euler  # numpy array [roll, pitch, yaw] in radians
            yaw = float(euler[2]) * 180.0 / 3.141592653589793
            pitch = float(euler[1]) * 180.0 / 3.141592653589793
            roll = float(euler[0]) * 180.0 / 3.141592653589793
            self.save_state_to_csv(state, yaw, pitch, roll, self.state_file_path, timestamp=ts)
            # Also log covariance matrix (EKF P)
            self.save_covariance_to_txt(kf.P, self.cov_file_path, timestamp=ts)

        # 3) IMU orientation
        imu = IMU()
        # IMU may be running in its own thread; get_euler returns (heading, roll, pitch)
        heading, roll, pitch = imu.get_euler()
        # get raw sensor measurements (accel, gyro, mag)
        accel = imu.get_accel()
        gyro = imu.get_gyro()
        mag = imu.get_mag()

        orient = SimpleNamespace(timestamp=ts, yaw=heading, pitch=pitch, roll=roll, accel=accel, gyro=gyro, mag=mag)
        # save orientation and raw sensor values
        self.save_orientation_to_csv(orient, self.imu_file_path)

        # 4) Encoder data
        encoder = BackWheelEncoder()
        # count = encoder.get_count()
        velocity = encoder.get_velocity()
        # self.save_encoder_to_csv(count, velocity, self.encoder_file_path, timestamp=ts)

        # 5) Path following data (motor speed and steering angle)
        if self.path_following.is_running():
            v_cmd, delta_cmd = self.path_following.get_current_commands()
            self.save_path_following_to_csv(v_cmd, delta_cmd, self.path_following_file_path, timestamp=ts)
    
    def end(self, interrupted):
        # Close all CSV files managed by the manager
        self.csv_manager.close_all()
        # Close covariance file separately (not managed by manager)
        if not self._cov_fh.closed:
            self._cov_fh.close()
    
    def is_finished(self):
        return False
    
    def save_uwb_pos_to_csv(self, position1: Optional[Position], position2: Optional[Position], filename: str, timestamp: Optional[float] = None) -> bool:
        """
        Save a position reading to a CSV file
        
        Args:
            position1: First Position object to save
            position2: Second Position object to save
            filename: CSV filename
            timestamp: Optional timestamp
            
        Returns:
            bool: True if successful, False otherwise
        """
        def _safe_get(p: Optional[Position], attr, default=''):
            return getattr(p, attr) if p is not None else default

        ts = timestamp if timestamp is not None else time.time()
        row = {
            'timestamp': ts,
            'x1': _safe_get(position1, 'x', ''),
            'y1': _safe_get(position1, 'y', ''),
            'z1': _safe_get(position1, 'z', ''),
            'quality1': _safe_get(position1, 'quality', ''),
            'x2': _safe_get(position2, 'x', ''),
            'y2': _safe_get(position2, 'y', ''),
            'z2': _safe_get(position2, 'z', ''),
            'quality2': _safe_get(position2, 'quality', ''),
        }
        
        # Try to get the file handle and writer from the manager; if not available, fallback to opening the file each time
        fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None))
        return write_csv_or_fallback(writer, fh, filename, self.UWB_FIELDNAMES, row)
    
    def save_orientation_to_csv(self, orientation, filename: str) -> bool:
        """
        Save IMU orientation reading to a CSV file
        
        Args:
            orientation: Orientation object with sensor data
            filename: CSV filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        ts = getattr(orientation, 'timestamp', time.time())

        def _triplet(val):
            if val is None:
                return ('', '', '')
            a, b, c = val
            return (a if a is not None else '', b if b is not None else '', c if c is not None else '')

        ax, ay, az = _triplet(getattr(orientation, 'accel', None))
        gx, gy, gz = _triplet(getattr(orientation, 'gyro', None))
        mx, my, mz = _triplet(getattr(orientation, 'mag', None))

        row = {
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
        }
        
        fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None)) if str(filename) in self.csv_manager.files else (None, None, None)
        return write_csv_or_fallback(writer, fh, filename, self.IMU_FIELDNAMES, row)

    def save_uwb_anchors_to_csv(self, anchors_info, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save UWB anchor info into a CSV file.

        anchors_info: expected to be the list returned by UWB.get_latest_anchor_info(),
                      i.e., a list of tuples (port, anchors) where anchors is a list of dicts
                      (each dict has keys 'name','id','position'=(x,y,z),'range').
        The CSV columns will be: timestamp, port, name, id, x, y, z, range
        """
        ts = timestamp if timestamp is not None else time.time()
        
        for port, anchors in anchors_info or []:
            if not anchors:
                row = {'timestamp': ts, 'port': port, 'name': '', 'id': '', 'x': '', 'y': '', 'z': '', 'range': ''}
            else:
                for anchor in anchors:
                    pos = anchor.get('position', (None, None, None))
                    row = {
                        'timestamp': ts,
                        'port': port,
                        'name': anchor.get('name', ''),
                        'id': anchor.get('id', ''),
                        'x': pos[0] if pos and len(pos) > 0 else '',
                        'y': pos[1] if pos and len(pos) > 1 else '',
                        'z': pos[2] if pos and len(pos) > 2 else '',
                        'range': anchor.get('range', ''),
                    }
                    fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None)) if str(filename) in self.csv_manager.files else (None, None, None)
                    write_csv_or_fallback(writer, fh, filename, self.ANCHORS_FIELDNAMES, row)
        return True

    def save_state_to_csv(self, state, yaw: float, pitch: float, roll: float, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save estimator state (pos, vel, euler) to CSV."""
        ts = timestamp if timestamp is not None else time.time()
        px, py, pz = state.pos
        vx, vy, vz = state.vel
        
        row = {
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
        }
        
        fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None)) if str(filename) in self.csv_manager.files else (None, None, None)
        return write_csv_or_fallback(writer, fh, filename, self.STATE_FIELDNAMES, row)

    def save_covariance_to_txt(self, P, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save covariance matrix in a readable text file.

        The file will contain a timestamp header followed by the matrix rows.
        Multiple calls append additional blocks (so the file contains history).
        """
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

    def save_encoder_to_csv(self, count: int, velocity: float, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save encoder count and velocity to CSV file.
        
        Args:
            count: Number of encoder counts
            velocity: Velocity in m/s
            filename: CSV filename
            timestamp: Optional timestamp
            
        Returns:
            bool: True if successful, False otherwise
        """
        ts = timestamp if timestamp is not None else time.time()
        row = {
            'timestamp': ts,
            'count': count,
            'velocity': velocity,
        }
        
        fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None)) if str(filename) in self.csv_manager.files else (None, None, None)
        return write_csv_or_fallback(writer, fh, filename, self.ENCODER_FIELDNAMES, row)

    def save_path_following_to_csv(self, motor_speed: float, steering_angle: float, filename: str, timestamp: Optional[float] = None) -> bool:
        """Save path following motor speed and steering angle to CSV file.
        
        Args:
            motor_speed: Motor speed in m/s
            steering_angle: Steering angle in radians
            filename: CSV filename
            timestamp: Optional timestamp
            
        Returns:
            bool: True if successful, False otherwise
        """
        ts = timestamp if timestamp is not None else time.time()
        steering_angle_deg = float(steering_angle) * 180.0 / np.pi
        
        row = {
            'timestamp': ts,
            'motor_speed_mps': motor_speed,
            'steering_angle_rad': steering_angle,
            'steering_angle_deg': steering_angle_deg,
        }
        
        fh, writer, _ = self.csv_manager.files.get(str(filename), (None, None, None)) if str(filename) in self.csv_manager.files else (None, None, None)
        return write_csv_or_fallback(writer, fh, filename, self.PATH_FOLLOWING_FIELDNAMES, row)