from Robot.Constants import Constants
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU
from Robot.Commands.LogDataCmd import LogDataCmd

class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.imu = IMU()
        self.uwb.start(uwb_tag_data=Constants.uwb_tag_data, anchors_pos=None)
                    
    def begin_data_log(self):
        LogDataCmd().schedule()

