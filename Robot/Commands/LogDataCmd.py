from structure.commands.Command import Command


import csv
import os
import time
from Robot.subsystems.sensors.UWBTag import Position

class LogDataCmd(Command):
    def __init__(self):
        super().__init__()
        
    def initalize(self):
        self.begin_timestamp = time.time()
    
    def execute(self):
        
        pass
    
    def end(self, interrupted):
        pass
    
    def is_finished(self):
        return False
    
    def save_uwb_pos_to_csv(self, position: Position, filename: str) -> bool:
        """
        Save a position reading to a CSV file
        
        Args:
            position: Position object to save
            filename: CSV filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(filename)
            
            # Open file in append mode
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'x', 'y', 'z', 'quality']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")
                                    
                # Write position data
                writer.writerow({
                    'timestamp': position.timestamp,
                    'x': position.x,
                    'y': position.y,
                    'z': position.z,
                    'quality': position.quality,
                })
                
            return True
            
        except Exception as e:
            print(f"Error saving position to CSV: {e}")
            return False
    
    def save_orientation_to_csv(self, orientation, filename: str) -> bool:
        """
        Save a position reading to a CSV file
        
        Args:
            position: Position object to save
            filename: CSV filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(filename)
            
            # Open file in append mode
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'yaw', 'pitch', 'roll']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {filename}")
                                    
                # Write position data
                writer.writerow({
                    'timestamp': orientation.timestamp,
                    'yaw': orientation.yaw,
                    'pitch': orientation.pitch,
                    'roll': orientation.roll,
                })
                
            return True
            
        except Exception as e:
            print(f"Error saving position to CSV: {e}")
            return False