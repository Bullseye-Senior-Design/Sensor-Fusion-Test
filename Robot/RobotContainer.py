from Robot.Constants import Constants
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.PathFollowing import PathFollowing
from Robot.subsystems.MotorControl import MotorControl

from Robot.subsystems.sensors.BackWheelEncoder import BackWheelEncoder
from Robot.subsystems.Clutches import Clutches
from Robot.Commands.LogDataCmd import LogDataCmd
from Robot.Commands.PlotStateCmd import PlotStateCmd
from Robot.Commands.AlignIMUToWorldCmd import AlignIMUToWorldCmd
from Robot.Commands.ZeroIMUCmd import ZeroIMUCmd
from Robot.Commands.MiniBullseyeControlCmd import MiniBullseyeControlCmd
from Robot.Commands.FollowPathCmd import FollowPathCmd


class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.back_Wheel_encoder = BackWheelEncoder()
        self.imu = IMU()
        self.clutches = Clutches()
        self.path_following = PathFollowing()
        self.motor_control = MotorControl()
        
         # Start subsystems
        self.clutches.start(left_clutch_pin=Constants.left_clutch_pin, right_clutch_pin=Constants.right_clutch_pin)
        self.uwb.start(uwb_tag_data=Constants.uwb_tag_data, anchors_pos=None)
        self.back_Wheel_encoder.start(pin=Constants.back_right_encoder_pin, active_high=True, pull_up=True, debounce_ms=10, edge='falling', wheel_circumference=Constants.wheel_circumference, counts_per_revolution=Constants.counts_per_revolution)
        
        #self.motor_control.default_command(MiniBullseyeControlCmd(self.motor_control, self.path_following))
        self.path_following.default_command(FollowPathCmd(self.motor_control, self.path_following))
                    
    def begin_data_log(self):
        LogDataCmd(self.path_following).schedule()
        PlotStateCmd().schedule()
        ZeroIMUCmd().schedule()
        
        # AlignIMUToWorldCmd(tau=0.5, duration=30.0).schedule()
                
    def shutdown(self):
        self.back_Wheel_encoder.stop()
        self.clutches.stop()
        self.uwb.stop_all()
        self.motor_control.stop()
