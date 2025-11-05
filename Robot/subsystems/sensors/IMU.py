# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
import threading
import board
import digitalio
import busio

import adafruit_bno055
import math
import numpy as np
import time
from typing import Optional, Tuple
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator 


class IMU():
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start()
        return cls._instance

    def _start(self):
        i2c = board.I2C() # uses board.SCL and board.SDA
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

        self.acceleration = (0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.magnetic = (0.0, 0.0, 0.0)
        self.quat = (0.0, 0.0, 0.0, 0.0)

        # Reference magnetometer vector in world frame (can be adjusted or updated later)
        self.mag_ref_world = np.array([0.0, 0.0, -1.0])

        self.interval = 0.01  # update interval in seconds
        self.mag_interval = self.interval * 5
        # timestamp of last magnetic sample
        self._last_mag_time = 0.0

        # Protect concurrent access from update thread and callers
        self._lock = threading.RLock()
        self.state_estimator = KalmanStateEstimator()

        # Start initial calibration in background (non-blocking) and then start continuous update
        # Calibration will write computed biases into the KalmanStateEstimator when done.
        t = threading.Thread(target=self.initial_calibration, kwargs={"duration": 1.0, "sample_delay": 0.01}, daemon=True)
        t.start()

        # Start magnetometer reference calibration in background (non-blocking)
        # runs as a daemon thread and returns immediately
        self.calibrate_mag_ref(duration=20.0, sample_delay=0.05, run_async=True)

        # Print any library-provided calibration info (if available)
        self.print_library_calibration()

        # Start continuous update loop immediately (calibration runs in parallel)
        self.begin()

    def get_gyro(self) -> tuple:
        with self._lock:
            return tuple(self.gyro)
    
    def get_accel(self) -> tuple:
        with self._lock:
            return tuple(self.acceleration)
    
    def get_mag(self) -> tuple:
        # units are microteslas
        with self._lock:
            return tuple(self.magnetic)

    def get_quat(self) -> tuple:
        with self._lock:
            return tuple(self.quat)
    
    def get_euler(self) -> tuple:
        with self._lock:
            q = tuple(self.quat)

        # ensure we have a valid quaternion (w, x, y, z)
        if not q or len(q) != 4:
            return (0.0, 0.0, 0.0)

        # ensure quaternion components are numeric (replace None with 0.0)
        def _safe(val):
            return 0.0 if val is None else float(val)

        # assume quaternion is (w, x, y, z) as returned by Adafruit BNO055
        w, x, y, z = (_safe(v) for v in q)

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if sinp >= 1.0:
            pitch = math.pi / 2
        elif sinp <= -1.0:
            pitch = -math.pi / 2
        else:
            pitch = math.asin(sinp)

        # yaw / heading (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # return in (heading, roll, pitch) degrees to match sensor.euler ordering
        deg = math.degrees
        return (deg(yaw), deg(roll), deg(pitch))

    def mag_interval_elapsed(self) -> bool:
        """Return True when it's time to sample the magnetometer.

        Does NOT update the timestamp; caller should update _last_mag_time
        after a successful read/assignment.
        """
        return (time.time() - self._last_mag_time) >= self.mag_interval
    
    def begin(self):
        def _update_loop():
            while True:
                self.update()
        
        threading.Thread(target=_update_loop, daemon=True).start()
        

    def _collect_samples_for_duration(self, read_fn, duration: float, sample_delay: float = 0.01):
        """Collect samples from read_fn for `duration` seconds and return a numpy array (N x 3)."""
        samples = []
        end_t = time.time() + duration
        while time.time() < end_t:
            try:
                v = read_fn()
            except Exception:
                v = None
            if v is not None and all(x is not None for x in v):
                try:
                    arr = np.asarray(v, dtype=float)
                    if arr.shape[0] >= 3:
                        samples.append(arr[0:3])
                except Exception:
                    pass
            time.sleep(sample_delay)
        if samples:
            return np.vstack(samples)
        else:
            return np.empty((0, 3), dtype=float)

    def _quat_to_rotmat_sensor(self, q_sensor: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Convert a quaternion in sensor order (w, x, y, z) to a 3x3 rotation matrix R (body -> world).
        Returns identity if input is invalid.
        """
        if q_sensor is None or len(q_sensor) != 4:
            return np.eye(3)
        w, x, y, z = (0.0 if v is None else float(v) for v in q_sensor)
        R = np.empty((3, 3))
        R[0, 0] = 1 - 2 * (y * y + z * z)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)
        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x * x + z * z)
        R[1, 2] = 2 * (y * z - x * w)
        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def calibrate_mag_ref(self, duration: float = 20.0, sample_delay: float = 0.05, normalize: bool = True, run_async: bool = True):
        """
        Calibrate `self.mag_ref_world` by collecting magnetometer samples and transforming them
        into the world frame using the sensor quaternion. Average the transformed vectors.

        - duration: seconds to collect samples (rotate device during this time to sample orientations)
        - sample_delay: seconds between samples
        - normalize: if True, store unit vector (direction only). The KF uses direction-only comparison.
        - async: if True, run calibration in a daemon thread and return the Thread object.

        Returns the Thread if async=True, otherwise None.
        """
        def _calibrate():
            print(f"IMU: starting mag_ref calibration for {duration:.1f}s - rotate the IMU on all axes...")
            samples = []
            end_t = time.time() + duration
            while time.time() < end_t:
                try:
                    mag = self.sensor.magnetic  # body-frame magnetometer reading
                    quat = self.sensor.quaternion  # sensor quaternion (w,x,y,z)
                except Exception:
                    mag = None
                    quat = None

                if mag is not None and all(v is not None for v in mag) and quat is not None and len(quat) == 4:
                    try:
                        mag_vec = np.asarray(mag, dtype=float)
                        R = self._quat_to_rotmat_sensor(quat)  # type: ignore # body -> world
                        mag_world = R @ mag_vec
                        samples.append(mag_world)
                    except Exception:
                        pass

                time.sleep(sample_delay)

            if not samples:
                print("IMU: mag_ref calibration collected no valid samples.")
                return

            arr = np.vstack(samples)
            mean_world = np.mean(arr, axis=0)

            if normalize:
                nrm = np.linalg.norm(mean_world)
                if nrm > 0:
                    mean_world = mean_world / nrm

            # thread-safe write
            with self._lock:
                self.mag_ref_world = mean_world.astype(float)
            print("IMU: mag_ref_world calibrated ->", self.mag_ref_world)

        if run_async:
            thread = threading.Thread(target=_calibrate, daemon=True)
            thread.start()
            return thread
        else:
            _calibrate()
            return None

    def print_library_calibration(self) -> None:
        """Read and print calibration information from the underlying sensor library (if present).

        This tries several common attributes/methods found on Adafruit BNO055 wrappers and
        prints whatever calibration/status/offset information is available. It is safe
        to call even if the sensor object doesn't expose these fields.
        """
        def begin():
            while not self.sensor.calibrated:
                time.sleep(0.1)
                print("IMU: waiting for sensor calibration...")
            
            accel_off = self.sensor.offsets_accelerometer
            gyro_off = self.sensor.offsets_gyroscope
            mag_off = self.sensor.offsets_magnetometer
            print("IMU: Library Calibration Offsets:")
            print("  Accelerometer offsets:", accel_off)
            print("  Gyroscope offsets:", gyro_off) 
            print("  Magnetometer offsets:", mag_off)
        
        
        threading.Thread(target=begin, daemon=True).start()

    def initial_calibration(self, duration: float, sample_delay: float):
        """
        Run a short calibration capturing gyro and accel while stationary (non-blocking).

        - duration: total seconds to sample (default 1.0)
        - sample_delay: seconds between sensor reads

        Results:
          - writes gyro bias into KalmanStateEstimator.bg (x[13:16])
          - writes accel bias (simple magnitude correction) into KalmanStateEstimator.ba (x[10:13])

        This function runs in a daemon thread (started from _start()) so it won't block the main loop.
        """
        try:
            print(f"IMU: starting initial calibration for {duration:.2f}s (stationary) in background...")
            # Collect gyro and accel samples directly from sensor
            gyro_samples = self._collect_samples_for_duration(lambda: self.sensor.gyro, duration, sample_delay)
            accel_samples = self._collect_samples_for_duration(lambda: self.sensor.acceleration, duration, sample_delay)

            gyro_bias = np.zeros(3, dtype=float)
            accel_bias = np.zeros(3, dtype=float)

            if gyro_samples.size:
                gyro_bias = np.mean(gyro_samples, axis=0)
                print("IMU: computed gyro bias (will set in estimator) =", gyro_bias)
            else:
                print("IMU: no gyro samples collected during calibration; gyro bias left zeros")

            if accel_samples.size:
                mean_acc = np.mean(accel_samples, axis=0)
                g_val = 9.80665
                mean_norm = np.linalg.norm(mean_acc)
                if mean_norm > 1e-6:
                    # simple magnitude correction: compute bias so corrected mean has magnitude g
                    scale = g_val / mean_norm
                    corrected = mean_acc * scale
                    accel_bias = mean_acc - corrected
                    print("IMU: computed accel bias (will set in estimator) =", accel_bias, " mean_acc=", mean_acc)
                else:
                    print("IMU: accel mean magnitude too small; accel bias left zeros")
            else:
                print("IMU: no accel samples collected during calibration; accel bias left zeros")

            # Write biases into KalmanStateEstimator using its public setter (thread-safe)
            try:
                # prefer using the estimator's API rather than touching internal arrays
                self.state_estimator.set_biases(accel_bias, gyro_bias)
                print("IMU: biases written to KalmanStateEstimator (ba, bg).")
            except Exception as e:
                print("IMU: failed to write biases to estimator:", e)
        except Exception as e:
            print("IMU: calibration exception:", e)

    def update(self):
        # continuous update loop; keep reads outside lock and assign under lock
        accel: Optional[Tuple[float, float, float]] = None
        gyro: Optional[Tuple[float, float, float]] = None
        magnetic: Optional[Tuple[float, float, float]] = None
        quat: Optional[Tuple[float, float, float, float]] = None
        
        accel_val = self.sensor.acceleration
        gyro_val = self.sensor.gyro
        mag_val = None
        if(self.mag_interval_elapsed()):
            try:
                mag_val = self.sensor.magnetic
            except Exception as e:
                print(f"IMU Readings - Mag: {e}")
        quat_val = self.sensor.quaternion
        # print(f"IMU Readings - Accel: {accel_val}, Gyro: {gyro_val}, Mag: {mag_val}, Quat: {quat_val}")

        # Use the pre-read values if available
        if all(v is not None for v in accel_val):
            accel = accel_val # type: ignore
        if all(v is not None for v in gyro_val):
            gyro = gyro_val # type: ignore
        if mag_val is not None and all(v is not None for v in mag_val):
            magnetic = mag_val # type: ignore
        if all(v is not None for v in quat_val):
            quat = quat_val # type: ignore

        # NOTE: biases are stored in the KalmanStateEstimator (x[10:13] = ba, x[13:16] = bg).
        # The estimator's predict() method subtracts those biases, so we pass raw measurements here.
        accel_arr: Optional[np.ndarray] = None
        gyro_arr: Optional[np.ndarray] = None
        if accel is not None:
            accel_arr = np.asarray(accel, dtype=float)
        if gyro is not None:
            gyro_arr = np.asarray(gyro, dtype=float)

        if accel_arr is not None and gyro_arr is not None:
            self.state_estimator.predict(accel_meas=accel_arr, gyro_meas=gyro_arr)
        if magnetic is not None:
            mag_arr = np.asarray(magnetic, dtype=float)
            # use instance mag_ref_world (set during _start, can be changed later)
            self.state_estimator.update_mag(mag_arr, self.mag_ref_world)

        with self._lock:
            if accel is not None:
                self.acceleration = tuple(accel)
            if gyro is not None:
                self.gyro = tuple(gyro)

            if magnetic is not None:
                self.magnetic = tuple(magnetic)
                # update last-mag timestamp only when we actually stored a magnetic sample
                self._last_mag_time = time.time()
            if quat is not None:
                self.quat = tuple(quat)

        # avoid busy loop
        time.sleep(self.interval)
             
    def end(self):
        pass