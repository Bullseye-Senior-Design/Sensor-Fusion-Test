
#!/usr/bin/env python3
"""
DWM1001-DEV Tag Position Reader (Low Latency Mode)

This program reads position data from a DWM1001-DEV Ultra-Wideband (UWB) tag.
Modified for "Saturation Polling" to detect position updates immediately 
and filter stale data.

Author: Generated for UWB Subsystem
Date: October 15, 2025 (Modified for Low Latency)
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
logger = logging.getLogger(f"UWBTag")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output

@dataclass
class Position:
    """Data class to store position information"""
    x: float
    y: float
    z: float
    quality: int
    timestamp: float

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        # Exact float comparison is intentional here. 
        # If the DWM1001 hasn't updated, the bytes in memory are identical.
        # If it HAS updated, UWB noise ensures at least one float will differ slightly.
        return self.x == other.x and self.y == other.y and self.z == other.z

@dataclass
class TagInfo:
    node_id: str
    anchors: Optional[List[Dict[str, Any]]] = None

@dataclass
class LocationData:
    anchors: Optional[List[Dict[str, Any]]]
    position: Optional[Position]

class UWBTag:
    """
    Class to handle communication with DWM1001-DEV tag and read position data
    """

    def __init__(self, port: str, anchors_pos_override: Optional[List[Tuple[int, float, float, float]]] = None, baudrate: int = 115200, timeout: float = 0.05, tag_offset: Optional[Tuple[float, float, float]] = None, interval: float = 0.1):
        """
        Initialize the DWM1001 reader in Low Latency Mode.
        
        Args:
            port: Serial port
            baudrate: 115200 default
            timeout: Serial read timeout. Kept low (0.05) for responsiveness.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.tag_offset = tag_offset
        self.anchors_pos_override = anchors_pos_override
        
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_reading = False
        self.tag_info = TagInfo(node_id="unknown")
        self.read_thread = None
        self.state_estimator = KalmanStateEstimator()
        
        self.position_lock = threading.RLock()
        self.last_position: Optional[Position] = None
        
        # Statistics for debugging
        self._stats_reads = 0
        self._stats_updates = 0

        atexit.register(self.disconnect)
        
    def connect(self) -> bool:
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                write_timeout=0.1
            )
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            self.is_connected = True
            logger.info("Connected to DWM1001 (Generic Mode)")
            return True
        except serial.SerialException as e:
            logger.error(f"Connect failed: {e}")
            return False

    def disconnect(self):
        self.stop_reading()
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("Disconnected")

    def _read_tlv_frame(self) -> Tuple[Optional[int], Optional[bytes]]:
        """Reads a TLV frame. Returns (Type, Value) or (None, None)."""
        if not self.serial_connection:
            return None, None
        
        try:
            # Read header (2 bytes)
            header = self.serial_connection.read(2)
            if len(header) < 2:
                return None, None
            
            t_byte = header[0]
            l_byte = header[1]
            
            if l_byte > 0:
                value = self.serial_connection.read(l_byte)
                if len(value) != l_byte:
                    return None, None
                return t_byte, value
            return t_byte, b''
        except serial.SerialException:
            return None, None

    def get_location_data(self) -> LocationData:
        """Sends request and reads response robustly."""
        if not self.serial_connection:
            return LocationData(None, None)
        
        # 2. Write 'dwm_loc_get' (0x0C)
        try:
            self.serial_connection.write(b'\x0C\x00')
        except Exception:
            return LocationData(None, None)

        pos_data = None
        
        # 3. Read loop: We wait slightly longer to ensure we capture the specific TLV we need
        # The DWM1001 sends multiple TLVs. We must loop until we find 0x41 or timeout.
        start_t = time.perf_counter()
        
        # We assume the loop cycle is faster than the serial baud rate transmission
        while (time.perf_counter() - start_t) < 0.02: 
            t, v = self._read_tlv_frame()
            
            if t is None or v is None:
                continue 

            # Type 0x41 is Position
            if t == 0x41 and len(v) >= 13: 
                x, y, z, qf = struct.unpack('<iiiB', v[:13])
                
                # Check for "invalid" zeros (Firmware couldn't solve position)
                if qf == 0:
                    # We continue looping here because we might want to clear the rest 
                    # of the buffer, or we just accept we got a failed frame.
                    return LocationData(None, None)
                
                pos_data = Position(
                    x=x/1000.0, 
                    y=y/1000.0, 
                    z=z/1000.0, 
                    quality=qf, 
                    timestamp=time.time()
                )
                
                logger.debug(f"Raw Position Data: x={pos_data.x:.3f}m, y={pos_data.y:.3f}m, z={pos_data.z:.3f}m, qf={pos_data.quality}")
                # Note: The 'Distances' packet (0x48) might still be coming.
                # It will stay in the OS serial buffer and be read as "garbage" 
                # (skipped) in the next loop iteration, which is fine.
                return LocationData(None, pos_data)
            
            # Type 0x40 is Error/Status
            elif t == 0x40:
                if len(v) > 0 and v[0] != 0:
                    # Error code returned (e.g. Busy)
                    return LocationData(None, None)

        return LocationData(None, None)

    def start_continuous_reading(self):
        """
        Starts the high-speed polling loop.
        It requests data constantly but only processes *changes* in position.
        """
        if self.is_reading: return
        self.is_reading = True
        
        def read_loop():
            logger.info("Starting high-speed position polling...")
            last_log_time = time.time()
            
            while self.is_reading:
                
                logger.debug(f"Time since last log: {time.time() - last_log_time:.2f}s")
                last_log_time = time.time()
                
                # 1. Get Data (Blocking call via serial, but fast)
                loc_data = self.get_location_data()
                
                if loc_data.position:
                    process_update = False
                    
                    with self.position_lock:
                        # 2. Check if data is STALE (Duplicate)
                        # We compare X, Y, Z. If they are identical to the last read,
                        # the tag has not updated its calculation yet.
                        logger.debug(f"Comparing positions: Last={self.last_position} vs New={loc_data.position}")
                        if (self.last_position is None) or (loc_data.position != self.last_position):
                            self.last_position = loc_data.position
                            process_update = True
                    
                    # 3. Only update EKF if the data is NEW
                    if process_update:
                        try:
                            meas = np.array([
                                loc_data.position.x, 
                                loc_data.position.y, 
                                loc_data.position.z
                            ])
                            if self.tag_offset:
                                self.state_estimator.update_uwb_range(meas, np.array(self.tag_offset), True)
                            else:
                                self.state_estimator.update_uwb_range(meas, use_offset= False)
                        except Exception as e:
                            logger.error(f"EKF Update Error: {e}")
                            
                # 5. NO SLEEP. 
                # We loop immediately to catch the next UART byte as soon as it arrives.
                # However, to prevent CPU melting if USB is disconnected, we do a tiny yield
                # if no data was found, but strictly 0 if we are getting data.
                if not loc_data.position:
                    time.sleep(0.1) 

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
    
    def stop_reading(self):
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
    
    def get_latest_position(self) -> Optional[Position]:
        with self.position_lock:
            return self.last_position
