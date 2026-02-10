from structure.commands.Command import Command
import tkinter as tk
from Robot.subsystems.MotorControl import MotorControl
from Robot.subsystems.PathFollowing import PathFollowing
from Robot.Commands.FollowPathCmd import FollowPathCmd


class MiniBullseyeControlCmd(Command):
    """Command that opens a non-blocking Tkinter GUI to control Mini Bullseye.

    The command remains active until the window is closed or the command
    is cancelled. GUI updates are handled in execute() so the scheduler
    can keep running. Uses MotorControl subsystem for motor commands.
    """

    def __init__(
        self,
        motor_control: MotorControl,
        path_following: PathFollowing,
        speed_min: int = -100,
        speed_max: int = 100,
        steer_min: int = -30,
        steer_max: int = 30,
        speed_step: int = 10,
        steer_step: int = 5,
        key_repeat_delay_ms: int = 80,
    ):
        super().__init__()
        self.add_requirement(motor_control)
        self.speed_min = int(speed_min)
        self.speed_max = int(speed_max)
        self.steer_min = int(steer_min)
        self.steer_max = int(steer_max)
        self.speed_step = int(speed_step)
        self.steer_step = int(steer_step)
        self.key_repeat_delay_ms = int(key_repeat_delay_ms)

        self._running = False
        
        # Get reference to motor control subsystem
        self.motor_control = motor_control
        self.path_following = path_following

        # GUI objects
        self.root = None
        self.status_label = None
        self.speed_slider = None
        self.steering_slider = None

        # keyboard state
        self.keys_pressed = {"w": False, "a": False, "s": False, "d": False}
        self._repeat_job = None

    def initialize(self):
        # Create GUI
        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"MiniBullseyeControlCmd: failed to create Tk root: {e}")
            self.root = None
            self._running = False
            return

        self.root.title("Mini Bullseye Control - WASD support")
        self.root.geometry("450x420")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status label
        self.status_label = tk.Label(self.root, text="Ready (use WASD or sliders)", font=("Arial", 11))
        self.status_label.pack(pady=8)

        # Speed Slider
        tk.Label(self.root, text="Speed (%)", font=("Arial", 12)).pack()
        self.speed_slider = tk.Scale(
            self.root,
            from_=self.speed_min,
            to=self.speed_max,
            orient=tk.HORIZONTAL,
            length=380,
            resolution=1,
            command=lambda x: self._update_and_send(),
        )
        self.speed_slider.pack(pady=5)
        self.speed_slider.set(0)

        # Steering Slider
        tk.Label(self.root, text="Steering Angle (°)", font=("Arial", 12)).pack()
        self.steering_slider = tk.Scale(
            self.root,
            from_=self.steer_min,
            to=self.steer_max,
            orient=tk.HORIZONTAL,
            length=380,
            resolution=1,
            command=lambda x: self._update_and_send(),
        )
        self.steering_slider.pack(pady=5)
        self.steering_slider.set(0)

        tk.Button(
            self.root,
            text="EMERGENCY STOP",
            font=("Arial", 14, "bold"),
            bg="red",
            fg="white",
            width=20,
            command=self._emergency_stop,
        ).pack(pady=20)

        # Bind keyboard events (works when window has focus)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_force()

        self._running = True

    def execute(self):
        if not self._running or self.root is None:
            return

        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._running = False

    def end(self, interrupted):
        if self._repeat_job is not None and self.root is not None:
            try:
                self.root.after_cancel(self._repeat_job)
            except Exception:
                pass
            self._repeat_job = None

        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

        if interrupted:
            print("MiniBullseyeControlCmd: interrupted")

    def is_finished(self):
        return not self._running

    def _on_close(self):
        self._running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def _send_data(self, speed, angle):
        """Send speed and angle commands via MotorControl subsystem.
        
        Args:
            speed: Speed percentage
            angle: Steering angle in degrees
        """
        self.motor_control.set_speed_angle(speed, angle)

    def _update_and_send(self):
        if self.speed_slider is None or self.steering_slider is None:
            return
        speed = self.speed_slider.get()
        angle = self.steering_slider.get()
        self._send_data(speed, angle)
        if self.status_label is not None:
            self.status_label.config(
                text=f"Speed: {speed:3d}%   |   Steering: {angle:3d}°",
                fg="black",
            )

    def _emergency_stop(self):
        if self.speed_slider is None or self.steering_slider is None:
            return
        self.speed_slider.set(0)
        self.steering_slider.set(0)
        self._update_and_send()
        if self.status_label is not None:
            self.status_label.config(text="EMERGENCY STOP – all zeroed", fg="red")

    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key == "o":
            self._schedule_follow_path()
            return
        if key in self.keys_pressed:
            self.keys_pressed[key] = True
            if self._repeat_job is not None and self.root is not None:
                try:
                    self.root.after_cancel(self._repeat_job)
                except Exception:
                    pass
                self._repeat_job = None
            self._repeat_action()

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in self.keys_pressed:
            self.keys_pressed[key] = False

    def _repeat_action(self):
        if self.speed_slider is None or self.steering_slider is None:
            return

        changed = False

        if self.keys_pressed["w"] and not self.keys_pressed["s"]:
            new_speed = min(self.speed_max, self.speed_slider.get() + self.speed_step)
            self.speed_slider.set(new_speed)
            changed = True
        elif self.keys_pressed["s"] and not self.keys_pressed["w"]:
            new_speed = max(self.speed_min, self.speed_slider.get() - self.speed_step)
            self.speed_slider.set(new_speed)
            changed = True

        if self.keys_pressed["a"] and not self.keys_pressed["d"]:
            new_angle = max(self.steer_min, self.steering_slider.get() - self.steer_step)
            self.steering_slider.set(new_angle)
            changed = True
        elif self.keys_pressed["d"] and not self.keys_pressed["a"]:
            new_angle = min(self.steer_max, self.steering_slider.get() + self.steer_step)
            self.steering_slider.set(new_angle)
            changed = True

        if changed:
            self._update_and_send()

        if any(self.keys_pressed.values()) and self.root is not None:
            self._repeat_job = self.root.after(self.key_repeat_delay_ms, self._repeat_action)

    def _schedule_follow_path(self):
        FollowPathCmd(self.motor_control, self.path_following).schedule()
        if self.status_label is not None:
            self.status_label.config(text="FollowPathCmd scheduled", fg="blue")
