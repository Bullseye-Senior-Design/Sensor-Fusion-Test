from Robot.Constants import Constants
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.PathFollowing import PathFollowing
from Robot.subsystems.DriveTrain import DriveTrain
import time

from Robot.subsystems.sensors.BackWheelEncoder import BackWheelEncoder
from Robot.subsystems.Clutches import Clutches
from Robot.Commands.LogDataCmd import LogDataCmd
from Robot.Commands.PlotStateCmd import PlotStateCmd
from Robot.Commands.AlignIMUToWorldCmd import AlignIMUToWorldCmd
from Robot.Commands.ZeroIMUCmd import ZeroIMUCmd
from Robot.Commands.FollowPathCmd import FollowPathCmd


class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.back_Wheel_encoder = BackWheelEncoder()
        self.imu = IMU()
        self.clutches = Clutches()
        self.path_following = PathFollowing()
        self.drive_train = DriveTrain()
        
         # Start subsystems
        self.clutches.start(left_clutch_pin=Constants.left_clutch_pin, right_clutch_pin=Constants.right_clutch_pin)
        self.uwb.start(uwb_tag_data=Constants.uwb_tag_data, anchors_pos=None)
        self.back_Wheel_encoder.start()
        
        #self.path_following.default_command(FollowPathCmd(self.drive_train, self.path_following))
                    
    def begin_data_log(self):
        LogDataCmd(self.path_following).schedule()
        ZeroIMUCmd(self.drive_train, self.path_following, schedule_followup=False).schedule()
        #PlotStateCmd().schedule()
        
        # AlignIMUToWorldCmd(tau=0.5, duration=30.0).schedule()
                
    def shutdown(self):
        self.back_Wheel_encoder.close()
        self.clutches.close()
        self.uwb.close_all()
        self.drive_train.stop()
        self.drive_train.close()