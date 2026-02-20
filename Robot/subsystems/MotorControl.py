"""Motor control subsystem for commanding robot motors via I2C."""

import smbus2
from structure.Subsystem import Subsystem
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.Constants import Constants
import math
import logging

logger = logging.getLogger(f"{__name__}.MotorControl")
logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output


class MotorControl(Subsystem):
    """Singleton subsystem for controlling robot motors via I2C.
    
    Handles communication with ESP32 to send speed and steering commands.
    """
    _instance = None
    
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize motor control subsystem."""
        self.i2c_bus_num = 1
        self.esp32_addr = 0x08
        
        self._bus = None
        
        # Current commanded values
        self._speed = 0
        self._angle = 0
        
        # Initialize I2C bus
        self._init_i2c()
        self.state_estimator = KalmanStateEstimator()
    
    def _init_i2c(self):
        """Initialize I2C bus connection."""
        try:
            self._bus = smbus2.SMBus(self.i2c_bus_num)
            logger.info(f"MotorControl: I2C bus {self.i2c_bus_num} initialized")
        except Exception as e:
            logger.error(f"MotorControl: I2C initialization failed: {e}")
            self._bus = None
    
    def set_speed_angle(self, speed: int, angle: int):
        """Set motor speed and steering angle.
        
        Args:
            speed: Speed percentage (-100 to 100)
            angle: Steering angle in degrees (-30 to 30)
        """
        self.state_estimator.set_steering_angle(math.radians(angle))
        self.state_estimator.set_rear_wheel_velocity(speed / 100.0 * Constants.rear_motor_top_speed)
        
        # Clamp to int16 range
        speed = max(-32768, min(32767, int(speed)))
        # flip angle sign to match expected convention (positive angle = left turn)
        angle = -max(-32768, min(32767, int(angle)))
        
        self._speed = speed
        self._angle = angle
        
        # Pack as 4 bytes: [speed_high, speed_low, angle_high, angle_low]
        data = [
            (speed >> 8) & 0xFF,
            speed & 0xFF,
            (angle >> 8) & 0xFF,
            angle & 0xFF,
        ]
        
        if self._bus:
            try:
                self._bus.write_i2c_block_data(self.esp32_addr, 0, data)
            except Exception as e:
                logger.error(f"MotorControl: I2C write failed: {e}")
                # Try to reinitialize I2C bus
                self._init_i2c()
    
    def get_current_commands(self):
        """Get current motor commands.
        
        Returns:
            tuple: (speed, angle)
        """
        return self._speed, self._angle
    
    def stop(self):
        """Stop motors (set speed and angle to zero)."""
        self.set_speed_angle(0, 0)
    
    def close(self):
        """Close I2C bus connection."""
        if self._bus is not None:
            try:
                self._bus.close()
                logger.info("MotorControl: I2C bus closed")
            except Exception:
                pass
            self._bus = None
