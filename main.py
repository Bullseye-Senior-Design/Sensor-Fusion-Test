from Debug import Debug
from Robot.Robot import Robot
from structure.RobotState import RobotState
from structure.Input.KeyboardListener import KeyboardListener
from structure.Input.ControllerListener import ControllerListener

import time

def main():
    
    robot = Robot()
    robot_state = RobotState()


    robot.robot_init()
    

    while True:        
        # Run periodic functions
        ControllerListener().update()
        KeyboardListener().update()      
        robot.robot_periodic()
        
        # Teleop (human operated) mode
        if robot_state.should_init_teleop():
            print("Teleop mode enabled")
            robot.teleop_init()
        
        if robot_state.is_teleop_enabled():
            robot.teleop_periodic()
        
        # Test mode
        if robot_state.should_init_test():
            print("Test mode enabled")
            robot.test_init()
        
        if robot_state.is_test_enabled():
            robot.test_periodic()
        
        # Disabled check
        if robot_state.should_init_disable():
            robot.disabled_init()
        
        # Add a small delay to prevent high CPU usage
        time.sleep(0.01)
        
if __name__ == "__main__":
    main()