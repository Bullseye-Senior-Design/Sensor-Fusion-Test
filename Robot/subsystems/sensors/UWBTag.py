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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

    def __init__(self, port: str, anchors_pos_override: Optional[List[Tuple[int, float, float, float]]] = None, baudrate: int = 115200, timeout: Optional[float] = None, tag_offset: Optional[Tuple[float, float, float]] = None):
        """
        Initialize the DWM1001 reader
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication baud rate (default: 115200)
            timeout: Serial read timeout in seconds (None blocks until data)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.anchors_pos_override = anchors_pos_override
        self.tag_offset = tag_offset
        
        self.interval = 0.1  # default read interval in seconds
        
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_reading = False
        self.tag_info = TagInfo(node_id="unknown")
        self.read_thread = None
        self.state_estimator = KalmanStateEstimator()
        
        self.position_lock = threading.RLock()
        self.last_position = None
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

    def _read_tlv_frame(self) -> Tuple[Optional[int], Optional[bytes]]:
        """
        Optimized TLV reader. Reads header in one go.
        """
        if not self.serial_connection:
            return None, None
        
        # Read 2 bytes (Type + Length) at once to reduce USB latency
        header = self.serial_connection.read(2)
        if len(header) < 2:
            return None, None
        
        t_byte = header[0]
        l_byte = header[1]
        
        if l_byte > 0:
            value = self.serial_connection.read(l_byte)
            if len(value) != l_byte:
                return None, None # Incomplete frame
            return t_byte, value
        
        return t_byte, b''

    def get_location_data(self) -> LocationData:
        if not self.serial_connection:
            return LocationData(None, None)

        # Send Request: dwm_loc_get (0x0C)
        try:
            self.serial_connection.write(b'\x0C\x00')
        except serial.SerialTimeoutException:
            logger.error("Serial write timeout")
            return LocationData(None, None)

        pos_data = None
        
        # We might receive multiple TLVs (Error code 0x40, then Position 0x41)
        # We loop briefly to find 0x41
        start_time = time.time()
        while (time.time() - start_time) < 0.05: # 50ms timeout for response
            t, v = self._read_tlv_frame()
            if t is None or v is None:
                break
            
            # 0x41 = Position Data
            if t == 0x41 and len(v) >= 13:
                x, y, z, qf = struct.unpack('<iiiB', v[:13])
                pos_data = Position(
                    x=x/1000.0, 
                    y=y/1000.0, 
                    z=z/1000.0, 
                    quality=qf, 
                    timestamp=time.time()
                )
                break # Found what we wanted
            
            # 0x40 = Error Code (Usually 0x00 = Success)
            elif t == 0x40:
                if len(v) > 0 and v[0] != 0:
                    # Non-zero error code
                    logger.warning(f"DWM1001 Error Code: {v[0]}")
                    pass

        return LocationData(None, pos_data)

    def start_continuous_reading(self):
        if self.is_reading: return
        self.is_reading = True
        
        def read_loop():
            # Local variables for speed
            last_log_time = time.time()
            count = 0
            
            while self.is_reading:
                start_time = time.time()
                
                loc_data = self.get_location_data()
                
                if loc_data.position:
                    with self.position_lock:
                        self.last_position = loc_data.position
                    
                    # Update EKF
                    try:
                        meas = np.array([loc_data.position.x, loc_data.position.y, loc_data.position.z])
                        self.state_estimator.update_uwb_range(meas)
                    except Exception:
                        pass
                    
                    count += 1

                # LOGGING: Only log once per second, NOT every frame
                now = time.time()
                if now - last_log_time > 1.0:
                    hz = count / (now - last_log_time)
                    if self.last_position:
                        logger.debug(f"Rate: {hz:.1f}Hz | Pos: ({self.last_position.x:.2f}, {self.last_position.y:.2f})")
                    else:
                        logger.debug(f"Rate: {hz:.1f}Hz | No Position Lock")
                    last_log_time = now
                    count = 0
                
                # 3. Rate Limiting
                elapsed = time.time() - start_time
                target_period = self.interval # 10Hz
                
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
    
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