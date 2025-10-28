# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
import adafruit_bno055
import board

# TODO fix later with new IMU code on Pi 5
class IMU():
    def __init__(self):
        
        i2c = board.I2C() # uses board.SCL and board.SDA
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def get_gyro(self):
        return self.sensor.gyro
    
    def get_accel(self):
        return self.sensor.acceleration
    
    def get_mag(self):
        return self.sensor.magnetic
    
    def get_euler_angle(self):
        return self.sensor.euler
    
    def get_quaternion(self):
        return self.sensor.quaternion 