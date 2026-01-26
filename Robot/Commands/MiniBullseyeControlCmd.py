import tkinter as tk
import smbus
import struct
from structure.commands.Command import Command

class MiniBullseyeControlCmd(Command):
    """
    Command to control Mini Bullseye robot via I2C using a tkinter GUI.
    Provides sliders for speed and steering angle control, plus emergency stop.
    """
    
    # ================= I2C Configuration =================
    I2C_BUS = 1
    ESP32_ADDR = 0x08
    
    def __init__(self):
        super().__init__()
        self.root = None
        self.speed_slider = None
        self.steering_slider = None
        self.status_label = None
        self.bus = None
        self.running = False
        
    def initialize(self):
        """Initialize I2C bus and create the GUI window."""
        try:
            self.bus = smbus.SMBus(self.I2C_BUS)
        except Exception as e:
            print(f"Failed to initialize I2C bus: {e}")
            return
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Mini Bullseye Control")
        self.root.geometry("400x300")
        
        # Speed Slider
        tk.Label(self.root, text="Speed (%)", font=("Arial", 12)).pack()
        self.speed_slider = tk.Scale(
            self.root,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda x: self._send_command()
        )
        self.speed_slider.pack()
        self.speed_slider.set(0)
        
        # Steering Slider
        tk.Label(self.root, text="Steering Angle (°)", font=("Arial", 12)).pack()
        self.steering_slider = tk.Scale(
            self.root,
            from_=-45,
            to=45,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda x: self._send_command()
        )
        self.steering_slider.pack()
        self.steering_slider.set(0)
        
        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=10)
        
        # Emergency Stop Button
        tk.Button(
            self.root,
            text="EMERGENCY STOP",
            font=("Arial", 12, "bold"),
            bg="red",
            fg="white",
            command=self._emergency_stop
        ).pack(pady=10)
        
        # Close Button
        tk.Button(
            self.root,
            text="Close Control",
            font=("Arial", 10),
            command=self._close_window
        ).pack(pady=5)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)
        
        self.running = True
    
    def execute(self):
        """Update the GUI - called repeatedly while command is running."""
        if self.root and self.running:
            try:
                self.root.update()
            except tk.TclError:
                # Window was closed
                self.running = False
    
    def end(self, interrupted):
        """Clean up resources when command ends."""
        # Send stop command
        if self.bus:
            try:
                self._send_stop_command()
            except Exception as e:
                print(f"Failed to send stop command: {e}")
        
        # Close GUI
        if self.root:
            try:
                self.root.destroy()
            except:
                pass
        
        # Close I2C bus
        if self.bus:
            try:
                self.bus.close()
            except:
                pass
        
        self.running = False
    
    def is_finished(self):
        """Command is finished when GUI window is closed."""
        return not self.running
    
    # ================= Private Methods =================
    
    def _send_command(self):
        """Send current angle and speed values to ESP32 via I2C."""
        if not self.bus:
            return
        
        angle = self.steering_slider.get()
        speed = self.speed_slider.get()
        
        # Pack: int16 (angle) + uint8 (speed)
        data = struct.pack('<hB', angle, speed)
        
        try:
            self.bus.write_i2c_block_data(self.ESP32_ADDR, 0, list(data))
            self.status_label.config(
                text=f"Sent → Angle: {angle}°, Speed: {speed}%",
                fg="green"
            )
        except Exception as e:
            self.status_label.config(text=f"I2C Error: {e}", fg="red")
    
    def _send_stop_command(self):
        """Send stop command (speed=0, angle=0) to ESP32."""
        if not self.bus:
            return
        
        data = struct.pack('<hB', 0, 0)
        try:
            self.bus.write_i2c_block_data(self.ESP32_ADDR, 0, list(data))
        except Exception as e:
            print(f"Failed to send stop command: {e}")
    
    def _emergency_stop(self):
        """Emergency stop - reset all controls to zero."""
        self.speed_slider.set(0)
        self.steering_slider.set(0)
        self._send_command()
    
    def _close_window(self):
        """Handle window close event."""
        self.running = False
