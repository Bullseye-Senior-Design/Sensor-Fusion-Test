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

        self.sensor.offsets_accelerometer = (-26, 32702, -50)
        self.sensor.offsets_gyroscope = (-1, -4, -1)
        self.sensor.offsets_magnetometer = (-839, -601, -413)

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

        # NOTE: biases are stored in the KalmanStateEstimator (x[10:13] = ba, x[13:16] = bg).
        # The estimator's predict() method subtracts those biases, so we pass raw measurements here.
        # accel_arr: Optional[np.ndarray] = None
        # gyro_arr: Optional[np.ndarray] = None
        # if accel is not None:
        #     accel_arr = np.asarray(accel, dtype=float)
        # if gyro is not None:
        #     gyro_arr = np.asarray(gyro, dtype=float)
        if quat is not None:
            quat_arr = np.asarray(quat, dtype=float)
            # convert sensor quaternion (w, x, y, z) to estimator order [qx,qy,qz,qw]
            try:
                q_est = MathUtil.quat_sensor_to_estimator(quat_arr)
            except Exception:
                q_est = quat_arr
            # update attitude in KalmanStateEstimator
            self.state_estimator.update_imu_attitude(q_meas=q_est)

        # if accel_arr is not None and gyro_arr is not None:
        #     self.state_estimator.predict(accel_meas=accel_arr, gyro_meas=gyro_arr)
        # if magnetic is not None:
        #     mag_arr = np.asarray(magnetic, dtype=float)
        #     # use instance mag_ref_world (set during _start, can be changed later)
        #     self.state_estimator.update_mag(mag_arr, self.mag_ref_world)

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