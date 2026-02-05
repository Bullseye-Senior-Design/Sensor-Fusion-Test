from structure.commands.Command import Command
import smbus2
import time
import numpy as np
from Robot.subsystems.Navigation import MPCNavigator


class FollowPathCmd(Command):
    """Command that uses MPCNavigator to follow a path.

    The command continuously polls the MPC navigation system and sends
    motor commands via I2C. Runs until the command is cancelled.
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        esp32_addr: int = 0x08,
        update_rate_hz: float = 20.0,
    ):
        """Initialize FollowPathCmd with a simple straight path.
        
        Args:
            i2c_bus: I2C bus number
            esp32_addr: I2C address of ESP32
            update_rate_hz: Rate at which to update motor commands
        """
        super().__init__()
        
        # Create a simple straight path: 10 meters forward along x-axis
        num_points = 100
        self.path_matrix = np.zeros((num_points, 3))
        self.path_matrix[:, 0] = np.linspace(0, 10, num_points)  # x from 0 to 10m
        self.path_matrix[:, 1] = 0.0  # y = 0
        self.path_matrix[:, 2] = 0.0  # theta = 0 (pointing forward)
        self.i2c_bus = int(i2c_bus)
        self.esp32_addr = int(esp32_addr)
        self.update_period = 1.0 / float(update_rate_hz)

        self._bus = None
        self._running = False
        self._last_update_time = 0.0
        
        # Get reference to navigator singleton
        self.navigator = MPCNavigator()

    def initialize(self):
        """Initialize I2C bus and start path following."""
        # Initialize I2C bus
        try:
            self._bus = smbus2.SMBus(self.i2c_bus)
            print("FollowPathCmd: I2C bus initialized")
        except Exception as e:
            print(f"FollowPathCmd: I2C initialization failed: {e}")
            self._bus = None

        # Set path and start navigation
        self.navigator.set_path(self.path_matrix)
        self.navigator.start_path_following()
        
        self._running = True
        self._last_update_time = time.time()
        print("FollowPathCmd: Path following initialized")

    def execute(self):
        """Poll navigation system and send motor commands."""
        if not self._running:
            return
        
        # Rate limiting
        current_time = time.time()
        if (current_time - self._last_update_time) < self.update_period:
            return
        
        self._last_update_time = current_time
        
        try:
            # Get current commands from navigator
            v_cmd, delta_cmd = self.navigator.get_current_commands()
            
            # Convert to motor commands
            # v_cmd is in m/s, delta_cmd is in radians
            # Convert velocity to percentage (assuming 1 m/s = 100%)
            speed_percent = int(v_cmd * 100.0)
            # Convert steering angle from radians to degrees
            angle_deg = int(np.degrees(delta_cmd))
            
            # Send to motors
            self._send_data(speed_percent, angle_deg)
            
        except Exception as e:
            print(f"FollowPathCmd: Error in execute: {e}")

    def end(self, interrupted):
        """Stop path following and clean up."""
        # Stop navigation
        self.navigator.stop_path_following()
        
        # Send zero commands
        if self._bus is not None:
            try:
                self._send_data(0, 0)
            except Exception:
                pass
        
        # Close I2C bus
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        
        self._running = False
        
        if interrupted:
            print("FollowPathCmd: interrupted")
        else:
            print("FollowPathCmd: completed")

    def is_finished(self):
        """Command runs until cancelled."""
        return not self._running

    def _send_data(self, speed, angle):
        """Send speed and angle commands via I2C.
        
        Args:
            speed: Speed percentage (-100 to 100)
            angle: Steering angle in degrees (-30 to 30)
        """
        # Clamp to int16 range
        speed = max(-32768, min(32767, int(speed)))
        angle = max(-32768, min(32767, int(angle)))

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
                print(f"FollowPathCmd: I2C write failed: {e}")
