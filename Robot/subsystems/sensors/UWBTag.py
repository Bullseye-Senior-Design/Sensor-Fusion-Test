
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
logger = logging.getLogger(f"{__name__}.UWBTag")
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
        """Sends request and reads response immediately."""
        if not self.serial_connection:
            return LocationData(None, None)

        # Write 'dwm_loc_get' (0x0C)
        try:
            self.serial_connection.write(b'\x0C\x00')
        except Exception:
            return LocationData(None, None)

        pos_data = None
        
        # Short loop to capture the response (expecting 0x41 or 0x40)
        # We don't wait long; we want to fail fast and retry if data is missing
        start_t = time.perf_counter()
        while (time.perf_counter() - start_t) < 0.05: 
            t, v = self._read_tlv_frame()
            if t is None or v is None:
                continue # Keep trying within timeout

            if t == 0x41 and len(v) >= 13: # Position Data
                x, y, z, qf = struct.unpack('<iiiB', v[:13])
                pos_data = Position(
                    x=x/1000.0, 
                    y=y/1000.0, 
                    z=z/1000.0, 
                    quality=qf, 
                    timestamp=time.time()
                )
                logger.debug(f"Received UWB Position: x={pos_data.x:.3f}m, y={pos_data.y:.3f}m, z={pos_data.z:.3f}m, quality={pos_data.quality}")
                # We found data, we can return immediately
                return LocationData(None, pos_data)
            
            elif t == 0x40: # Error/Status
                if len(v) > 0 and v[0] != 0:
                    # If error is legitimate, stop trying this frame
                    break

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
                self._stats_reads += 1
                
                # 1. Get Data (Blocking call via serial, but fast)
                loc_data = self.get_location_data()
                
                if loc_data.position:
                    process_update = False
                    
                    with self.position_lock:
                        # 2. Check if data is STALE (Duplicate)
                        # We compare X, Y, Z. If they are identical to the last read,
                        # the tag has not updated its calculation yet.
                        if (self.last_position is None) or (loc_data.position != self.last_position):
                            self.last_position = loc_data.position
                            process_update = True
                            self._stats_updates += 1
                    
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

                # 4. Reporting (1Hz)
                now = time.time()
                if now - last_log_time > 1.0:
                    with self.position_lock:
                        current = self.last_position
                    
                    hz_poll = self._stats_reads / (now - last_log_time)
                    hz_update = self._stats_updates / (now - last_log_time)
                    
                    log_msg = f"Poll: {hz_poll:.0f}Hz | Fresh Updates: {hz_update:.1f}Hz"
                    if current:
                        log_msg += f" | Pos: ({current.x:.2f}, {current.y:.2f})"
                    
                    logger.debug(log_msg)
                    
                    last_log_time = now
                    self._stats_reads = 0
                    self._stats_updates = 0
                
                # 5. NO SLEEP. 
                # We loop immediately to catch the next UART byte as soon as it arrives.
                # However, to prevent CPU melting if USB is disconnected, we do a tiny yield
                # if no data was found, but strictly 0 if we are getting data.
                if not loc_data.position:
                    time.sleep(0.0001) 

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
    
    def stop_reading(self):
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
    
    def get_latest_position(self) -> Optional[Position]:
        with self.position_lock:
            return self.last_position
