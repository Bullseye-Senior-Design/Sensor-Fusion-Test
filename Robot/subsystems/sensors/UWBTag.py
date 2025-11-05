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
import time
import threading
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
        
    def connect(self) -> bool:
        """
        Establish serial connection to DWM1001-DEV tag
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Test connection by sending shell command
            self.serial_connection.write(b'\r')
            time.sleep(0.5)
            self.serial_connection.write(b'\r')
            time.sleep(0.5)
            
            self.is_connected = True
            logger.info(f"Successfully connected to DWM1001-DEV on {self.port}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection"""
        self.stop_reading()
        if self.serial_connection and self.serial_connection.is_open:
            # Exit shell mode before closing
            try:
                self.serial_connection.write(b'quit\r\n')
                time.sleep(0.5)
            except:
                pass
            self.serial_connection.close()
            logger.info("Disconnected from DWM1001-DEV")
        self.is_connected = False
    
    def get_location_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Position]]:
        """
        Get current position from the tag
        
        Returns:
            Tuple of (anchors list or None, Position object or None)
        """
        if not self.is_connected or not self.serial_connection:
            logger.error("Not connected to DWM1001-DEV")
            return None, None
        
        try:
            line = self.serial_connection.readline()    

            response = line.decode('utf-8', errors='ignore').strip()    
            
            # Parse position data
            # Expected output may contain DIST...ANx... and POS,x,y,z,quality
            if 'POS' in response or 'DIST' in response:
                anchors, position = self._parse_position(response)
                return anchors, position
            else:
                # logger.warning(f"No position data in response: {response}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None, None
    
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

                anchors.append({
                    'name': name,
                    'id': anchor_id,
                    'position': (ax, ay, az),
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
    
    def start_continuous_reading(self, interval: float = 0.1, debug: bool = False):
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
        self.serial_connection.write(b'lec\r')  # Get Raw Distance Measurements
        time.sleep(1)

        def read_loop():
            while self.is_reading:
                anchors, position = self.get_location_data()

                # store anchors if provided (for diagnostics/visualization only)
                if anchors:
                    self.tag_info.anchors = anchors
                    if debug:
                        self.print_anchor_info()

                # Use fused POS (world) position for EKF update instead of per-anchor ranges
                if position:
                    with self.position_lock:
                        self.tag_info.position = position

                    # Build measurement vector and optional tag offset
                    tag_pos_meas = np.array([position.x, position.y, position.z], dtype=float)
                    tag_offset_vec = None if self.tag_offset is None else np.array(self.tag_offset, dtype=float)

                    try:
                        self.state_estimator.update_uwb_range(tag_pos_meas, tag_offset=tag_offset_vec)
                    except Exception as e:
                        logger.debug(f"EKF UWB POS update skipped: {e}")

                    if debug:
                        self.print_position(position)

                time.sleep(interval) # TODO check if needed
        
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