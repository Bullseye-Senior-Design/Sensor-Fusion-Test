from Robot.Constants import Constants
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU
from Robot.subsystems.sensors.Encoder import Encoder
from Robot.Commands.LogStateCmd import LogDataCmd
from Robot.Commands.PlotStateCmd import PlotStateCmd
from Robot.Commands.AlignIMUCmd import AlignIMUCmd
from Robot.Commands.ZeroIMUCmd import ZeroIMUCmd

class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.encoder = Encoder()
        self.imu = IMU()
        self.uwb.start(uwb_tag_data=Constants.uwb_tag_data, anchors_pos=None)
        self.encoder.start(pin=Constants.back_right_encoder_pin, active_high=True, pull_up=True, debounce_ms=1, edge='rising')
                    
    def begin_data_log(self):
        LogDataCmd().schedule()
        PlotStateCmd().schedule()
        ZeroIMUCmd().schedule()