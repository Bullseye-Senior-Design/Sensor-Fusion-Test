from typing import List, Tuple
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
