import math
from typing import List, Tuple, Optional, Dict, Any
from .UWBTag import UWBTag, Position
import logging
import threading
import time
from Robot.Constants import UWBTagInfo

logger = logging.getLogger(__name__)


class UWB:
    """
    UWB Subsystem to manage one or more `UWBTag` readers.

    Usage:
      - Pass a list of serial port names: UWB(['COM3','COM4'])

    The constructor will create the requested `UWBTag` instances and, by default,
    attempt to connect and start continuous reading on each.
    """
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(
        self,
        uwb_tag_data: List[UWBTagInfo],
        anchors_pos: List[Tuple[int, float, float, float]] | None,
        baudrate: int = 115200,
        timeout: float = 1.0,
        interval: float = 0.1,
        start_immediately: bool = True,
    ):
        self.interval = interval
        
        self.anchors_pos = anchors_pos

        # Use Any here so static checkers don't require resolving UWBTag symbols
        self.tags: List[UWBTag] = []
        for tag_info in uwb_tag_data:
            tag = UWBTag(port=tag_info.port, anchors_pos_override=anchors_pos, baudrate=baudrate, timeout=timeout, tag_offset=tag_info.offset)
            self.tags.append(tag)

        if start_immediately:
            self.bootup()
    
    def get_positions(self) -> List[Position]:
        """Get latest position from all connected tags."""
        positions: List[Position] = []
        for tag in self.tags:
            pos = tag.get_latest_position()
            if pos is not None:
                positions.append(pos)
        return positions
    
    def get_angle(self) -> float | None:
        """Compute robot heading (radians) corrected for tag offsets.

        Returns robot yaw in radians ([-pi, pi]). Requires at least two tags.
        """
        positions = self.get_positions()
        # need at least two tags to compute heading
        if len(positions) < 2:
            return None
        
        # instantaneous vector between tags in world frame
        dx = positions[1].x - positions[0].x
        dy = positions[1].y - positions[0].y
        yaw_inst = math.atan2(dy, dx)  # heading of tag-to-tag vector (world)

        # read tag offsets (body-frame) from UWBTag objects, fallback to zero
        off0 = getattr(self.tags[0], "tag_offset", None) or (0.0, 0.0, 0.0)
        off1 = getattr(self.tags[1], "tag_offset", None) or (0.0, 0.0, 0.0)

        # delta offset in body frame
        dx_b = float(off1[0]) - float(off0[0])
        dy_b = float(off1[1]) - float(off0[1])

        # offset angle in body frame (angle of vector from tag0->tag1 in robot coords)
        offset_angle = math.atan2(dy_b, dx_b)

        # robot yaw = measured yaw - offset_angle
        robot_yaw = yaw_inst - offset_angle

        # normalize to [-pi, pi]
        robot_yaw = (robot_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return robot_yaw

    def connect_all(self) -> List[Tuple[str, bool]]:
        """Attempt to connect all tags. Returns list of (port, success)."""
        results: List[Tuple[str, bool]] = []
        for tag in self.tags:
            try:
                ok = tag.connect()
            except Exception as e:
                logger.exception(f"Exception connecting to {tag.port}: {e}")
                ok = False
            results.append((tag.port, ok))
        return results

    def bootup(self, retry_interval: float = 2.0, max_retries: int = 3) -> None:
        """Start a background thread that connects and starts tags with retries.

        This will return immediately. The boot thread is a daemon so it won't
        prevent process exit.
        """
        if hasattr(self, '_boot_thread') and self._boot_thread and self._boot_thread.is_alive():
            logger.warning('Bootup already running')
            return

        stop_event = threading.Event()
        self._boot_stop_event = stop_event

        def _boot():
            for tag in self.tags:
                retries = 0
                while retries < max_retries and not stop_event.is_set():
                    try:
                        if not tag.is_connected:
                            ok = tag.connect()
                        else:
                            ok = True
                        if ok:
                            tag.start_continuous_reading(self.interval)
                            break
                    except Exception:
                        logger.exception(f"Boot: error starting {tag.port}")
                    retries += 1
                    time.sleep(retry_interval)

        t = threading.Thread(target=_boot, daemon=True)
        self._boot_thread = t
        t.start()

    def stop_bootup(self) -> None:
        """Signal a running bootup thread to stop (if any)."""
        if hasattr(self, '_boot_stop_event') and self._boot_stop_event:
            self._boot_stop_event.set()

    def stop_all(self) -> None:
        """Stop reading and disconnect all tags."""
        for tag in self.tags:
            try:
                tag.stop_reading()
            except Exception:
                logger.exception(f"Failed to stop reading on {tag.port}")
            try:
                tag.disconnect()
            except Exception:
                logger.exception(f"Failed to disconnect {tag.port}")

    def __len__(self) -> int:
        return len(self.tags)

    def get_latest_anchor_info(self) -> List[Tuple[str, Optional[List[Dict[str, Any]]]]]:
        """Return the latest anchor info for all tags.

        Returns a list of tuples (port, anchors) where anchors is a shallow copy
        of the anchors list from the tag or None if no anchors are available.
        """
        results: List[Tuple[str, Optional[List[Dict[str, Any]]]]] = []
        for tag in self.tags:
            try:
                anchors = tag.get_latest_anchor_info()
            except Exception:
                logger.exception(f"Error reading anchors from {tag.port}")
                anchors = None
            results.append((tag.port, anchors))
        return results
