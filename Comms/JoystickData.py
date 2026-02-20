from dataclasses import dataclass
@dataclass 
class JoystickData:
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    Dpad_up: bool
    Dpad_down: bool
    Dpad_left: bool
    Dpad_right: bool
    Btn_A: bool
    Btn_B: bool
    Btn_X: bool
    Btn_Y: bool
    Btn_LB: bool
    Btn_RB: bool
    Btn_LS: bool
    Btn_RS: bool
    Btn_R2: bool
    Btn_L2: bool
    Btn_Share: bool
    Btn_Options: bool
