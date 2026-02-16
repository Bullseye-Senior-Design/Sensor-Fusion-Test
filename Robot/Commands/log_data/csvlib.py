"""Utility functions for CSV logging operations."""

import csv
import os
from typing import Optional, Dict, Any, Tuple


def setup_csv_file(filepath: str, fieldnames: list) -> Tuple[Any, Any, bool]:
    """Open or create a CSV file with headers.
    
    Args:
        filepath: Path to the CSV file
        fieldnames: List of column names
        
    Returns:
        Tuple of (file_handle, csv_writer, is_new_file)
            where is_new_file indicates if headers were written
    """
    try:
        file_exists = os.path.exists(filepath)
        is_new = file_exists and os.path.getsize(filepath) == 0
        
        fh = open(filepath, 'a', newline='')
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        
        if not file_exists or is_new:
            writer.writeheader()
            return fh, writer, True
        return fh, writer, False
    except Exception as e:
        print(f"Error setting up CSV file {filepath}: {e}")
        raise


def write_csv_row(writer: csv.DictWriter, file_handle: Any, row: Dict[str, Any]) -> bool:
    """Write a row to CSV and flush the file.
    
    Args:
        writer: csv.DictWriter instance
        file_handle: Open file handle
        row: Dictionary of values to write
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        writer.writerow(row)
        file_handle.flush()
        return True
    except Exception as e:
        print(f"Error writing CSV row: {e}")
        return False


def close_csv_file(file_handle: Any) -> bool:
    """Safely close a CSV file handle.
    
    Args:
        file_handle: File handle to close
        
    Returns:
        bool: True if successful or already closed, False on error
    """
    try:
        if file_handle is not None and not file_handle.closed:
            file_handle.close()
        return True
    except Exception as e:
        print(f"Error closing CSV file: {e}")
        return False


def write_csv_or_fallback(writer: Optional[csv.DictWriter], 
                          file_handle: Optional[Any],
                          filename: str,
                          fieldnames: list,
                          row: Dict[str, Any]) -> bool:
    """Write to CSV using persistent writer, or fallback to opening file each time.
    
    This is a helper for methods that want to use persistent file handles when available
    but need a fallback for compatibility.
    
    Args:
        writer: Persistent csv.DictWriter, or None to use fallback
        file_handle: Persistent file handle, or None to use fallback
        filename: Path to CSV file
        fieldnames: List of column names
        row: Dictionary of values to write
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Try to use persistent writer first
    if writer is not None and file_handle is not None:
        return write_csv_row(writer, file_handle, row)
    
    # Fallback: open file each time
    try:
        filename = str(filename)
        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as csvfile:
            temp_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                temp_writer.writeheader()
            temp_writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error writing to CSV (fallback): {e}")
        return False


class CSVFileManager:
    """Manager for multiple CSV files with automatic setup and cleanup."""
    
    def __init__(self):
        """Initialize the CSV file manager."""
        self.files = {}  # dict mapping file_path -> (file_handle, writer)
    
    def setup_file(self, filepath: str, fieldnames: list) -> Tuple[Any, Any]:
        """Setup a CSV file and store the handle and writer.
        
        Args:
            filepath: Path to the CSV file
            fieldnames: List of column names
            
        Returns:
            Tuple of (file_handle, csv_writer)
        """
        if filepath not in self.files:
            fh, writer, _ = setup_csv_file(filepath, fieldnames)
            self.files[filepath] = (fh, writer, fieldnames)
        return self.files[filepath][0], self.files[filepath][1]
    
    def write_row(self, filepath: str, row: Dict[str, Any]) -> bool:
        """Write a row to a managed CSV file.
        
        Args:
            filepath: Path to the CSV file
            row: Dictionary of values to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath not in self.files:
            print(f"File {filepath} not managed. Call setup_file() first.")
            return False
        
        fh, writer, _ = self.files[filepath]
        return write_csv_row(writer, fh, row)
    
    def close_all(self):
        """Close all managed CSV files."""
        for filepath, (fh, writer, _) in self.files.items():
            if not close_csv_file(fh):
                print(f"Warning: Failed to close {filepath}")
        self.files.clear()
    
    def close_file(self, filepath: str) -> bool:
        """Close a specific managed CSV file.
        
        Args:
            filepath: Path to the CSV file to close
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath in self.files:
            fh, _, _ = self.files.pop(filepath)
            return close_csv_file(fh)
        return False
