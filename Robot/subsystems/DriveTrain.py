"""Motor control subsystem for commanding robot motors via I2C."""

from structure.Subsystem import Subsystem
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.Constants import Constants
from Robot.subsystems.driveTrainChildren.DAC import DAC
from Robot.subsystems.driveTrainChildren.FrontWheelEncoder import FrontWheelEncoder

import math
import logging
import spidev
import RPi.GPIO as GPIO

logger = logging.getLogger(f"{__name__}.DriveTrain")
logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output


class DriveTrain(Subsystem):
    
    def __init__(self):
        """Initialize motor control subsystem."""
        try:
            self._dac = DAC()
            self._front_encoder = FrontWheelEncoder()
            self._backwheel_forward_ssr_pin = Constants.backwheel_forward_ssr_pin
            self._backwheel_reverse_ssr_pin = Constants.backwheel_reverse_ssr_pin
            self._backwheel_power_ssr_pin = Constants.backwheel_power_ssr_pin
            self._frontwheel_power_ssr_pin = Constants.frontwheel_power_ssr_pin
            self._dac_backwheel_channel = Constants.dac_backwheel_channel
            self._dac_frontwheel_channel = Constants.dac_frontwheel_channel
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._backwheel_forward_ssr_pin, GPIO.OUT)
            GPIO.setup(self._backwheel_reverse_ssr_pin, GPIO.OUT)
            GPIO.setup(self._backwheel_power_ssr_pin, GPIO.OUT)
            GPIO.setup(self._frontwheel_power_ssr_pin, GPIO.OUT)
            
            GPIO.output(self._backwheel_power_ssr_pin, GPIO.HIGH)
            GPIO.output(self._frontwheel_power_ssr_pin, GPIO.HIGH)

            logger.info(f"DriveTrain SPI initialized on bus {Constants.spi_bus}, device {Constants.dac_spi_device}, mode {Constants.dac_spi_mode}")
        
        except Exception as e:
            logger.error(f"Failed to initialize SPI for DriveTrain: {e}")
        
        self._angle = 90 # Default to straight
        self._speed = 0 # Default to stopped
    
    def set_speed_angle(self, speed: float, angle: float):
        """Set motor speed and angle.
        speed: -1.0 to 1.0 (negative for reverse)
        angle: 0 to 180 (90 is straight, <90 left, >90 right)
        """
        self._speed = speed
        self._angle = angle

        is_reverse = speed < 0
        self._set_wheel_directions(is_reverse)
        self._dac.write_dac(self._dac_backwheel_channel, abs(speed))
        self._dac.write_dac(self._dac_frontwheel_channel, abs(speed))

    def _set_wheel_directions(self, is_reverse: bool):
        """Set SSRs for wheel direction based on speed sign."""
        if not is_reverse:
            # Forward
            GPIO.output(self._backwheel_forward_ssr_pin, GPIO.HIGH)
            GPIO.output(self._backwheel_reverse_ssr_pin, GPIO.LOW)
        else:
            # Reverse
            GPIO.output(self._backwheel_forward_ssr_pin, GPIO.LOW)
            GPIO.output(self._backwheel_reverse_ssr_pin, GPIO.HIGH)
            
    def disengage_frontwheel(self):
        GPIO.output(self._frontwheel_power_ssr_pin, GPIO.LOW)

    def engage_frontwheel(self):
        GPIO.output(self._frontwheel_power_ssr_pin, GPIO.HIGH)
        
    def disengage_backwheel(self):
        GPIO.output(self._backwheel_power_ssr_pin, GPIO.LOW)
        
    def engage_backwheel(self):
        GPIO.output(self._backwheel_power_ssr_pin, GPIO.HIGH)

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
        try:
            self._dac.close()
            self._front_encoder.close()
            GPIO.cleanup()
        except Exception:
            pass
                        

