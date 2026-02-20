from dataclasses import dataclass
@dataclass 
class PathData:
    position_list: list[tuple[float, float]]
    id: int