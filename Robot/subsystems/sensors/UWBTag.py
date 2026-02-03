#!/usr/bin/env python3
"""
DWM1001-DEV Tag Position Reader

This program reads position data from a DWM1001-DEV Ultra-Wideband (UWB) tag.
The DWM1001-DEV communicates via UART/Serial interface and provides real-time
location data in a RTLS (Real Time Location System) network.

Requirements:
- pyserial library for serial communication
- DWM1001-DEV tag configured as a tag in RTLS network
- Proper COM port connection

Author: Generated for UWB Subsystem
Date: October 15, 2025
"""

import serial
import struct
import time
import threading
import atexit
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import logging
import numpy as np
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Data class to store position information"""
    x: float
    y: float
    z: float
    quality: int
    timestamp: float

@dataclass
class TagInfo:
    """Data class to store tag information"""
    node_id: str
    position: Optional[Position] = None
    battery_level: Optional[int] = None
    update_rate: Optional[int] = None
    anchors: Optional[List[Dict[str, Any]]] = None

@dataclass
class LocationData:
    """Return type for location data reads."""
    anchors: Optional[List[Dict[str, Any]]]
    position: Optional[Position]

class UWBTag:
    """
    Class to handle communication with DWM1001-DEV tag and read position data
    """

    def __init__(self, port: str, anchors_pos_override: Optional[List[Tuple[int, float, float, float]]] = None, baudrate: int = 115200, timeout: float = 1.0, tag_offset: Optional[Tuple[float, float, float]] = None):
        """
        Initialize the DWM1001 reader
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication baud rate (default: 115200)
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.anchors_pos_override = anchors_pos_override
        self.tag_offset = tag_offset
        
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_reading = False
        self.tag_info = TagInfo(node_id="unknown")
        self.read_thread = None
        self.state_estimator = KalmanStateEstimator()
        
        self.position_lock = threading.RLock()
        # Ensure best-effort cleanup on interpreter exit
        try:
            atexit.register(self.disconnect)
        except Exception:
            # registration failure shouldn't block normal operation
            logger.debug("Failed to register atexit disconnect handler")
        
    def connect(self) -> bool:
        """Establish connection in Generic (TLV) Mode"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )

            # Default is Generic mode; reset buffers
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()

            self.is_connected = True
            logger.info("Connected in TLV (Generic) Mode")
            return True

        except serial.SerialException as e:
            logger.error(f"Connect failed: {e}")
            return False

    
    def disconnect(self):
        """Close the serial connection"""
        self.stop_reading()
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
            finally:
                logger.debug("Serial connection closed")
            logger.info("Disconnected from DWM1001-DEV")
        self.is_connected = False

    def request_location_tlv(self):
        """Sends the dwm_loc_get request (Type=0x0C, Length=0x00)."""
        # TLV request format is: [Type][Length][Value...]
        # For dwm_loc_get there is no payload, so Length = 0x00.
        cmd = bytearray([0x0C, 0x00])
        if self.serial_connection:
            self.serial_connection.write(cmd)

    def _read_tlv_frame(self):
        """Helper to read a single TLV block from the serial stream."""
        # TLV response framing:
        # - 1 byte Type
        # - 1 byte Length
        # - N bytes Value (N == Length)
        if not self.serial_connection:
            return None, None
        
        t_byte = self.serial_connection.read(1)
        if not t_byte:
            return None, None

        l_byte = self.serial_connection.read(1)
        if not l_byte:
            return None, None

        length = ord(l_byte)
        value = self.serial_connection.read(length)
        return ord(t_byte), value   
    
    def get_location_data(self) -> LocationData:
        """Get current position from the tag using TLV parsing."""
        if not self.is_connected or not self.serial_connection:
            logger.error("Not connected to DWM1001-DEV")
            return LocationData(None, None)

        # Request a TLV response frame from the tag.
        self.request_location_tlv()

        pos_data: Optional[Position] = None
        anchor_list: List[Dict[str, Any]] = []

        # The response to dwm_loc_get is a sequence of TLV blocks. We read a
        # handful of blocks and extract the ones we care about:
        #   0x40: Error code
        #   0x41: Position (x, y, z, qf)
        #   0x48/0x49: Anchor info / distances (not yet parsed here)
        for _ in range(5):
            t, v = self._read_tlv_frame()
            if t is None:
                continue

            # Position TLV payload is 13 bytes (little-endian):
            #   x(int32), y(int32), z(int32), qf(uint8)
            # DWM1001-DEV reports position in millimeters, convert to meters
            if t == 0x41 and v and len(v) >= 13:
                x, y, z, qf = struct.unpack('<iiiB', v[:13])
                pos_data = Position(x=x/1000.0, y=y/1000.0, z=z/1000.0, quality=qf, timestamp=time.time())
                if pos_data != self.get_latest_position():
                    logger.debug(f"UWBTag: Parsed same position TLV: x={pos_data.x:.3f}, y={pos_data.y:.3f}, z={pos_data.z:.3f}, qf={pos_data.quality}")

            elif t == 0x48:
                # Distance Info (Anchor distances) - parsing not implemented
                pass

        return LocationData(anchor_list if anchor_list else None, pos_data)
    
    def _parse_position(self, response: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Position]]:
        """
        Parse position data from DWM1001 response
        
        Args:
            response: Raw response string from DWM1001
            
        Returns:
            Position object or None if parsing fails
        """
        try:
            # Normalize and split tokens
            parts = [p.strip() for p in response.replace('\n', '').split(',') if p is not None and p.strip() != '']
            n = len(parts)
            idx = 0

            anchors: List[Dict[str, Any]] = []

            # If message starts with DIST, skip it and optional count
            if idx < n and parts[idx].upper() == 'DIST':
                idx += 1
                # optional count token
                if idx < n:
                    try:
                        int(parts[idx])
                        idx += 1
                    except ValueError:
                        # not a count, continue
                        pass

            # Parse anchor blocks: expect sequence of ANx, id, ax, ay, az, range
            while idx < n and parts[idx].upper().startswith('AN'):
                if idx + 5 >= n:
                    break
                name = parts[idx]
                anchor_id = parts[idx + 1]
                try:
                    ax = float(parts[idx + 2])
                    ay = float(parts[idx + 3])
                    az = float(parts[idx + 4])
                    rng = float(parts[idx + 5])
                except ValueError:
                    # malformed anchor block -> stop parsing anchors
                    break

                # Override anchor position if anchor_pos_override is provided
                anchor_position = (ax, ay, az)
                if self.anchors_pos_override is not None:
                    for override_id, override_x, override_y, override_z in self.anchors_pos_override:
                        if int(anchor_id) == override_id:
                            anchor_position = (override_x, override_y, override_z)
                            break
                            
                anchors.append({
                    'name': name,
                    'id': anchor_id,
                    'position': anchor_position,
                    'range': rng,
                })

                idx += 6

            # Look for POS token and parse the estimated position
            pos_x = pos_y = pos_z = None
            quality = 0
            while idx < n:
                if parts[idx].upper() == 'POS' and idx + 3 < n:
                    try:
                        pos_x = float(parts[idx + 1])
                        pos_y = float(parts[idx + 2])
                        pos_z = float(parts[idx + 3])
                        if idx + 4 < n:
                            # quality may be an int or float-like string
                            try:
                                quality = int(float(parts[idx + 4]))
                            except ValueError:
                                quality = 0
                    except ValueError:
                        pos_x = pos_y = pos_z = None
                    break
                idx += 1

            # attach anchors to tag_info for external use
            self.tag_info.anchors = anchors if anchors else None

            if pos_x is None or pos_y is None or pos_z is None:
                # no POS in message; return anchors and no position
                return (anchors if anchors else None), None

            position = Position(
                x=pos_x,
                y=pos_y,
                z=pos_z,
                quality=quality,
                timestamp=time.time()
            )

            return (anchors if anchors else None), position

        except Exception as e:
            logger.error(f"Error parsing position data: {e}")
            return None, None
    
    def start_continuous_reading(self, interval: float = 0.05, debug: bool = False):
        """
        Start continuous position reading in a separate thread
        
        """
        if self.is_reading:
            logger.warning("Already reading continuously")
            return
        
        self.is_reading = True
        if not self.serial_connection:
            logger.error("Serial connection not established")
            return
        
        self.serial_connection.reset_input_buffer()

        def read_loop():
            try: 
                while self.is_reading:
                    location = self.get_location_data()
                    anchors = location.anchors
                    position = location.position

                    # store anchors if provided (for diagnostics/visualization only)
                    if anchors:
                        self.tag_info.anchors = anchors
                        if debug:
                            self.print_anchor_info()

                    # logger.info(f"UWBTag: Read position data {position}")

                    # Use fused POS (world) position for EKF update instead of per-anchor ranges
                    if position:
                        with self.position_lock:
                            self.tag_info.position = position

                        # Build measurement vector and optional tag offset
                        tag_pos_meas = np.array([position.x, position.y, position.z], dtype=float)
                        tag_offset_vec = None if self.tag_offset is None else np.array(self.tag_offset, dtype=float)

                        # EKF update    
                        try:
                            self.state_estimator.update_uwb_range(tag_pos_meas, tag_offset=tag_offset_vec)
                        except Exception as e:
                            logger.debug(f"EKF UWB POS update skipped: {e}")

                        if debug:
                            self.print_position(position)

                    time.sleep(interval) # TODO check if needed
            except KeyboardInterrupt:
                # Graceful exit on Ctrl-C
                self.disconnect()
                return
        
        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
        
        
        logger.info("Started continuous position reading")
    
    def stop_reading(self):
        """Stop continuous position reading"""
        self.is_reading = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        logger.info("Stopped continuous position reading")
    
    def get_latest_position(self) -> Optional[Position]:
        """Get the latest position reading"""
        with self.position_lock:
            return self.tag_info.position

    def get_latest_anchor_info(self) -> Optional[List[Dict[str, Any]]]:
        """Return the latest anchor information in a thread-safe way.

        Returns a shallow copy of the anchors list (or None) while holding
        the internal lock to avoid races with the read thread.
        """
        with self.position_lock:
            anchors = self.tag_info.anchors
            if anchors is None:
                return None
            # return a shallow copy so callers cannot mutate internal state
            return [a.copy() for a in anchors]
        
    def print_anchor_info(self):
        """Print current anchor information"""
        if not self.tag_info.anchors:
            logger.info("No anchor information available")
            return
        
        for anchor in self.tag_info.anchors:
            pos = anchor['position']
            logger.info(f"Anchor {anchor['name']} (ID: {anchor['id']}): "
                        f"Position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                        f"Range={anchor['range']:.3f}m")

    def print_position(self, position: Position):
        """
        Callback function to handle new position data
        
        Args:
            position: Position object with current location data
        """
        logger.info(f"Position: X={position.x:.3f}m, Y={position.y:.3f}m, Z={position.z:.3f}m, "
            f"Quality={position.quality}, Time={position.timestamp:.2f}")