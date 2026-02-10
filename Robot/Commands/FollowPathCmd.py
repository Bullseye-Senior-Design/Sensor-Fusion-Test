from structure.commands.Command import Command
import time
import numpy as np
from Robot.subsystems.PathFollowing import PathFollowing
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.MotorControl import MotorControl
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FollowPathCmd(Command):
    """Command that uses MPCNavigator to follow a path.

    The command continuously polls the MPC navigation system and sends
    motor commands to MotorControl subsystem. Runs until the command is cancelled.
    """

    def __init__(
        self,
        motor_control: MotorControl,
        path_following: PathFollowing,
    ):
        """Initialize FollowPathCmd with a simple straight path.
        
        Args:
            update_rate_hz: Rate at which to update motor commands
        """
        super().__init__()
        self.motor_control = motor_control
        self.path_following = path_following
        self.add_requirement(motor_control)
        self.add_requirement(path_following)
        
        # Get current position from EKF
        state_estimator = KalmanStateEstimator()
        current_state = state_estimator.get_state()
        start_x, start_y = current_state.pos[0], current_state.pos[1]
        start_yaw = state_estimator.euler[2]
        
        # Create a simple straight path: 10 meters forward from current position
        num_points = 100
        self.path_matrix = np.zeros((num_points, 3))
        # Path goes forward in the direction of current yaw
        self.path_matrix[:, 0] = start_x + np.linspace(0, 10, num_points) * np.cos(start_yaw)
        self.path_matrix[:, 1] = start_y + np.linspace(0, 10, num_points) * np.sin(start_yaw)
        self.path_matrix[:, 2] = start_yaw  # Keep same heading
        

        self._running = False
        self._last_update_time = 0.0
        
    def initialize(self):
        """Start path following."""
        # Set path and start navigation
        self.path_following.set_path(self.path_matrix)
        self.path_following.start_path_following()
        
        self._running = True
        self._last_update_time = time.time()
        logger.info("FollowPathCmd: Path following initialized")

    def execute(self):
        """Poll navigation system and send motor commands."""
        if not self._running:
            return
                
        try:
            # Get current commands from navigator
            v_cmd, delta_cmd = self.path_following.get_current_commands()
            
            # Convert to motor commands
            # v_cmd is in m/s, delta_cmd is in radians
            # Convert velocity to percentage (assuming 1 m/s = 100%)
            speed_percent = int(v_cmd * 100.0)
            # Convert steering angle from radians to degrees
            angle_deg = int(np.degrees(delta_cmd))
            
            logger.debug(f"FollowPathCmd: v_cmd={v_cmd:.2f} m/s, delta_cmd={delta_cmd:.2f} rad -> speed={speed_percent}%, angle={angle_deg} deg")
            
            # Send to motors via MotorControl subsystem
            self.motor_control.set_speed_angle(speed_percent, angle_deg)
            
        except Exception as e:
            logger.info(f"FollowPathCmd: Error in execute: {e}")

    def end(self, interrupted):
        """Stop path following and clean up."""
        # Stop navigation
        self.path_following.stop_path_following()
        
        # Stop motors
        self.motor_control.stop()
        
        self._running = False
        
        if interrupted:
            print("FollowPathCmd: interrupted")
        else:
            print("FollowPathCmd: completed")

    def is_finished(self):
        """Command runs until cancelled."""
        return not self._running
