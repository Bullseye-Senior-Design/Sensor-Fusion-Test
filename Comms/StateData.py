from typing import Optional
from enum import Enum 
from dataclasses import dataclass
class State(Enum):
    DISABLED = 0
    AUTONOMOUS = 1
    TELEOP = 2
    TEST = 3
@dataclass
class StateData:
    state: State
    path_speed: Optional[float]
    path_id: Optional[int]