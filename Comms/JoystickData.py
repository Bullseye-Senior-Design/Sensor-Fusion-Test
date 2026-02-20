from dataclasses import dataclass
@dataclass 
class JoystickData:
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    dpad_up: bool
    dpad_down: bool
    dpad_left: bool
    dpad_right: bool
    btn_A: bool
    btn_B: bool
    btn_X: bool
    btn_Y: bool
    btn_LB: bool
    btn_RB: bool
    btn_LS: bool
    btn_RS: bool
    btn_R2: bool
    btn_L2: bool
    btn_share: bool
    btn_options: bool
