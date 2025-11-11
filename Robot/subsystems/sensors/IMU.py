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
from Robot.MathUtil import MathUtil

# implement a simple low-pass IIR filter for smoothing IMU data
# cutoff frequency fc_hz, sampling frequency fs_hz
class _LowPassIIR:
    def __init__(self, fc_hz: float, fs_hz: float, y0: float = 0.0):
        dt = 1.0 / fs_hz
        RC = 1.0 / (2.0 * math.pi * fc_hz)
        self.alpha = dt / (RC + dt)
        self.y = float(y0)
    def update(self, x: float) -> float:
        # differential equation implementation: (y[n] = y[n−1] + α (x[n] − y[n−1]))
        self.y += self.alpha * (x - self.y)
        return self.y

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
        self.quat = (0.0, 0.0, 0.0, 1.0)

        self.interval = 0.01  # update interval in seconds
        self.mag_interval = self.interval * 5
        # Filters (tune fc as needed)
        fs_hz = 1.0 / self.interval # interval in herts
        self._accel_lpf = [_LowPassIIR(fc_hz=8.0,  fs_hz=fs_hz) for _ in range(3)]
        
        # timestamp of last magnetic sample
        self._last_mag_time = 0.0

        # Protect concurrent access from update thread and callers
        self._lock = threading.RLock()
        self.state_estimator = KalmanStateEstimator()

        # yaw offset to apply to sensor attitude (radians). This rotates the
        # sensor-reported orientation into the chosen world frame.
        self._yaw_offset_rad = 0.0
        # cached estimator-order quaternion (aligned) for quick access
        self._quat_est_cached = (0.0, 0.0, 0.0, 1.0)
        self._is_offset_set = False

        self.sensor.offsets_accelerometer = (-26, 0, -50)
        self.sensor.offsets_gyroscope = (-1, -4, -1)
        self.sensor.offsets_magnetometer = (-839, -601, -413)
        
        print("IMU offset values: {}, {}, {}".format(
            self.sensor.offsets_accelerometer,
            self.sensor.offsets_gyroscope,
            self.sensor.offsets_magnetometer))

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
    
    def set_yaw_offset(self, yaw_offset_deg: float) -> None:
        """Set a yaw offset (degrees). The offset is stored and applied to
        the quaternion sent to the estimator and available via
        `get_aligned_quat()`.
        """
        yaw_deg = float(yaw_offset_deg)
        with self._lock:
            self._yaw_offset_rad = math.radians(yaw_deg)
            self._is_offset_set = True

    def get_aligned_quat(self) -> tuple:
        """Return the last quaternion adjusted by the configured yaw offset.

        Returned ordering is estimator ordering [qx,qy,qz,qw]. If no quaternion
        has been read yet this returns a unit quaternion.
        """
        with self._lock:
            raw = self.quat
            if not raw or len(raw) != 4:
                return (0.0, 0.0, 0.0, 1.0)
            try:
                q_sensor = np.asarray(raw, dtype=float)
                q_est = MathUtil.quat_sensor_to_estimator(q_sensor)
            except Exception:
                q_est = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            # apply yaw offset if present
            if self._is_offset_set and abs(self._yaw_offset_rad) > 1e-12:
                q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, self._yaw_offset_rad]))
                q_est = MathUtil.quat_mul(q_yaw, q_est)
                q_est = MathUtil.quat_normalize(q_est)
            return tuple(q_est)
        
    
    def get_euler(self) -> tuple:
        """Return the current (yaw, roll, pitch) in degrees, adjusted by yaw offset.
        """
        # Use MathUtil helpers: convert sensor quaternion ordering to estimator
        # ordering, apply yaw offset if set, then convert to Euler using
        # MathUtil.quat_to_euler which returns (roll, pitch, yaw) in radians.
        with self._lock:
            raw = tuple(self.quat)
            yaw_off = float(self._yaw_offset_rad)

        # validate
        if not raw or len(raw) != 4:
            return (0.0, 0.0, 0.0)

        q_sensor = np.asarray(raw, dtype=float)
        q_est = MathUtil.quat_sensor_to_estimator(q_sensor)

        # apply yaw offset if configured
        if self._is_offset_set and abs(yaw_off) > 1e-12:
            q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, yaw_off]))
            q_est = MathUtil.quat_mul(q_yaw, q_est)
            q_est = MathUtil.quat_normalize(q_est)

        # MathUtil.quat_to_euler returns [roll, pitch, yaw] in radians
        rpy = MathUtil.quat_to_euler(q_est)

        roll = float(rpy[0])
        pitch = float(rpy[1])
        yaw = float(rpy[2])

        deg = math.degrees
        # return in (heading, roll, pitch) degrees to match previous API
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
        

    def print_library_calibration(self) -> None:
        """Read and print calibration information from the underlying sensor library (if present).

        This tries several common attributes/methods found on Adafruit BNO055 wrappers and
        prints whatever calibration/status/offset information is available. It is safe
        to call even if the sensor object doesn't expose these fields.
        """
        def begin():
            while not self.sensor.calibrated:
                time.sleep(0.1)
            
            accel_off = self.sensor.offsets_accelerometer
            gyro_off = self.sensor.offsets_gyroscope
            mag_off = self.sensor.offsets_magnetometer
            print("IMU: Library Calibration Offsets:")
            print("  Accelerometer offsets:", accel_off)
            print("  Gyroscope offsets:", gyro_off) 
            print("  Magnetometer offsets:", mag_off)
        
        
        threading.Thread(target=begin, daemon=True).start()


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

        # Apply low-pass filter (if samples present)
        if accel is not None:
            ax = [self._accel_lpf[i].update(float(accel[i])) for i in range(3)]
            accel = (ax[0], ax[1], ax[2])

        if quat is not None:
            quat_arr = np.asarray(quat, dtype=float)
            # convert sensor quaternion (w, x, y, z) to estimator order [qx,qy,qz,qw]
            q_est = MathUtil.quat_sensor_to_estimator(quat_arr)
            # apply configured yaw offset before sending to estimator
            with self._lock:
                yaw_off = float(self._yaw_offset_rad)
            if self._is_offset_set:
                q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, yaw_off]))
                q_est = MathUtil.quat_mul(q_yaw, q_est)
                q_est = MathUtil.quat_normalize(q_est)
                # update attitude in KalmanStateEstimator
                self.state_estimator.update_imu_attitude(q_meas=q_est)

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
                # also keep a cached estimator-order aligned quaternion for quick access
                q_sensor = np.asarray(quat, dtype=float)
                q_est_cached = MathUtil.quat_sensor_to_estimator(q_sensor)
                if self._is_offset_set:
                    q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, self._yaw_offset_rad]))
                    q_est_cached = MathUtil.quat_mul(q_yaw, q_est_cached)
                    q_est_cached = MathUtil.quat_normalize(q_est_cached)
                    self._quat_est_cached = tuple(q_est_cached)


        # avoid busy loop
        time.sleep(self.interval)
             
    def end(self):
        pass