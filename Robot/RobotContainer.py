

from structure.Input.KeyboardInput import KeyboardInput
from structure.Input.KeyboardListener import KeyboardListener
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from subsystems.sensors.UWB import UWB

class RobotContainer:
    def __init__(self):
        self.uwb = UWB()
        self.uwb.start(ports=['/dev/ttyUSB0'])
                    
    def teleop_init(self):
        pass