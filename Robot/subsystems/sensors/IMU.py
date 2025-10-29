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


class MagnetometerReference:
    """Simple magnetometer reference helper.

    Provides basic calibration (offsets and scale) and heading calculation.

    Usage:
      mag = MagnetometerReference(offsets=(0,0,0), scales=(1,1,1))
      cal = mag.apply(raw_mag_tuple)
      heading = mag.heading(raw_mag_tuple, accel_tuple=None)

    The heading() method returns degrees in range [0, 360).
    If an accelerometer sample is provided (ax,ay,az), a tilt-compensated
    heading is attempted. Otherwise a simple 2D heading atan2(my, mx) is used.
    """
    def __init__(self, offsets=(0.0, 0.0, 0.0), scales=(1.0, 1.0, 1.0)):
        self.offsets = tuple(float(x) for x in offsets)
        self.scales = tuple(float(s) for s in scales)

    def calibrate(self, offsets=None, scales=None):
        """Set calibration parameters.

        offsets: (ox, oy, oz) to subtract from raw readings
        scales: (sx, sy, sz) to divide calibrated values by
        """
        if offsets is not None:
            self.offsets = tuple(float(x) for x in offsets)
        if scales is not None:
            self.scales = tuple(float(s) for s in scales)

    def apply(self, raw_mag):
        """Apply calibration to a raw (mx, my, mz) tuple and return (mx, my, mz)."""
        try:
            mx, my, mz = raw_mag
        except Exception:
            raise ValueError("raw_mag must be a 3-tuple")
        ox, oy, oz = self.offsets
        sx, sy, sz = self.scales
        # avoid division by zero
        sx = sx if sx != 0.0 else 1.0
        sy = sy if sy != 0.0 else 1.0
        sz = sz if sz != 0.0 else 1.0
        return ((mx - ox) / sx, (my - oy) / sy, (mz - oz) / sz)

    def heading(self, raw_mag, accel=None):
        """Compute heading in degrees [0,360).

        If accel (ax,ay,az) is provided, perform tilt compensation. If tilt
        compensation math fails for any reason, fall back to simple 2D heading.
        """
        mx, my, mz = self.apply(raw_mag)

        # simple 2D heading (no tilt compensation)
        def simple_heading(mx, my):
            h = math.degrees(math.atan2(my, mx))
            if h < 0:
                h += 360.0
            return h

        if accel is None:
            return simple_heading(mx, my)

        try:
            ax, ay, az = accel
            # compute roll and pitch from accelerometer
            roll = math.atan2(ay, az)
            pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

            # tilt compensation
            Xh = mx * math.cos(pitch) + mz * math.sin(pitch)
            Yh = (mx * math.sin(roll) * math.sin(pitch)
                  + my * math.cos(roll)
                  - mz * math.sin(roll) * math.cos(pitch))

            h = math.degrees(math.atan2(Yh, Xh))
            if h < 0:
                h += 360.0
            return h
        except Exception:
            return simple_heading(mx, my)


class IMU():
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        
        i2c = board.I2C() # uses board.SCL and board.SDA
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

        self.acceleration = (0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.magnetic = (0.0, 0.0, 0.0)
        self.quat = (0.0, 0.0, 0.0, 0.0)
        
        self.interval = 0.01  # update interval in seconds
        self.mag_interval = self.interval * 5
        # timestamp of last magnetic sample
        self._last_mag_time = 0.0
        
        # Protect concurrent access from update thread and callers
        self._lock = threading.RLock()
        self.state_estimator = KalmanStateEstimator()
        
        self.begin()

    def get_gyro(self) -> tuple:
        with self._lock:
            return tuple(self.gyro)
    
    def get_accel(self) -> tuple:
        with self._lock:
            return tuple(self.acceleration)
    
    def get_mag(self) -> tuple:
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
        
        threading.Thread(target=self.update, daemon=True).start()
        

    def update(self):
        # continuous update loop; keep reads outside lock and assign under lock
            accel: Optional[Tuple[float, float, float]] = None
            gyro: Optional[Tuple[float, float, float]] = None
            magnetic: Optional[Tuple[float, float, float]] = None
            quat: Optional[Tuple[float, float, float, float]] = None
            
            # Read sensor attributes defensively: catching Remote I/O (or other) errors
            accel_val = self.sensor.acceleration
            gyro_val = self.sensor.gyro
            mag_val = self.sensor.magnetic
            quat_val = self.sensor.quaternion

            # Use the pre-read values if available
            if all(v is not None for v in accel_val):
                accel = accel_val # type: ignore
            if all(v is not None for v in gyro_val):
                gyro = gyro_val # type: ignore
            # only poll magnetic sensor at lower rate
            if self.mag_interval_elapsed() and all(v is not None for v in mag_val):
                magnetic = mag_val # type: ignore
            if all(v is not None for v in quat_val):
                quat = quat_val # type: ignore

            if accel is not None and gyro is not None:
                accel_arr = np.asarray(accel, dtype=float)
                gyro_arr = np.asarray(gyro, dtype=float)
                self.state_estimator.predict(accel_meas=accel_arr, gyro_meas=gyro_arr)
            if magnetic is not None:
                mag_arr = np.asarray(magnetic, dtype=float)
                mag_ref_world = np.array([0.0, 0.0, 1.0])
                self.state_estimator.update_mag(mag_arr, mag_ref_world)

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