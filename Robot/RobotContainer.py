

from structure.Input.KeyboardInput import KeyboardInput
from structure.Input.KeyboardListener import KeyboardListener
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sensors.UWB import UWB
from Robot.subsystems.sensors.IMU import IMU
from Robot.Commands.LogStateCmd import LogDataCmd
from Robot.Commands.PlotStateCmd import PlotStateCmd

class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.imu = IMU()
        self.uwb.start(ports=['/dev/ttyACM0', '/dev/ttyACM1'], anchors_pos=None)
                    
    def begin_data_log(self):
        LogDataCmd().schedule()
        PlotStateCmd().schedule()

