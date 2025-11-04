from structure.commands.Command import Command

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib import transforms as mtransforms
import matplotlib.image as mpimg  # <--- Added for image loading
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.sensors.IMU import IMU
import os


class PlotStateCmd(Command):
    """Command that plots the EKF position (x,y) in a Matplotlib figure embedded
    in a non-blocking Tkinter window with a rotating dump truck image.
    """

    def __init__(self, max_points: int = 1000):
        super().__init__()
        self.max_points = max_points
        self._running = False

        # plot state
        self.xs = []
        self.ys = []

        # GUI objects
        self.root = None
        self.figure = None
        self.ax = None
        self.line = None
        self.canvas = None
        
        # top-down widgets
        self.ax_top = None
        self.truck_image = None
        self.truck_artist = None
        self.yaw_text = None

        # estimator and sensors
        self.estimator = KalmanStateEstimator()
        self.imu = IMU()

    def initialize(self):
        if tk is None:
            print("PlotStateCmd: tkinter not available; cannot create GUI.")
            self._running = False
            return

        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"PlotStateCmd: failed to create Tk root: {e}")
            self.root = None
            self._running = False
            return

        self.root.wm_title("EKF Position Plot")

        # Use two subplots: main XY plot on left and a top-down yaw view
        self.figure = Figure(figsize=(8, 4), dpi=100)
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        # main XY plot
        self.ax = self.figure.add_subplot(gs[0, 0])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        self.line, = self.ax.plot([], [], "b.-", markersize=4)

        # top-down yaw view
        self.ax_top = self.figure.add_subplot(gs[0, 1])
        self.ax_top.set_title("Top-down (yaw)")
        self.ax_top.set_xlim(-1.5, 1.5)
        self.ax_top.set_ylim(-1.5, 1.5)
        self.ax_top.set_aspect("equal")
        self.ax_top.axis("off")

        # --- Load the truck image ---
        try:
            img_path = os.path.join(os.getcwd(), "dump_truck.png")
            self.truck_image = mpimg.imread(img_path)
        except Exception as e:
            print(f"PlotStateCmd: failed to load dump_truck.png: {e}")
            self.truck_image = np.ones((100, 100, 3))  # fallback white square

        # Center the image around origin (so it rotates in place)
        extent = [-0.75, 0.75, -0.75, 0.75]
        self.truck_artist = self.ax_top.imshow(
            self.truck_image,
            origin="upper",
            extent=extent,
            zorder=5
        )

        # yaw text
        self.yaw_text = self.ax_top.text(
            0.0, -1.2, "Yaw: --°", ha="center", va="center", fontsize=10
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._running = True

        # close handler
        self.root.protocol("WM_DELETE_WINDOW", self.end)

    def execute(self):
        if not self._running or self.root is None:
            return

        # Get EKF position
        try:
            pos = self.estimator.pos
            x = float(pos[0])
            y = float(pos[1])
        except Exception:
            x, y = np.nan, np.nan

        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > self.max_points:
            self.xs = self.xs[-self.max_points :]
            self.ys = self.ys[-self.max_points :]

        # Update line
        self.line.set_data(self.xs, self.ys)
        self.ax.relim()
        self.ax.autoscale_view()

        try:
            self.canvas.draw_idle()
            self.canvas.draw()
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._running = False
            return

        # Update yaw (rotation)
        try:
            euler = self.imu.get_euler()
            yaw = float(euler[2])
            deg = np.degrees(yaw)
        except Exception:
            yaw, deg = 0.0, float("nan")

        # rotate truck image
        if self.truck_artist is not None:
            base_transform = self.ax_top.transData
            rot = mtransforms.Affine2D().rotate(yaw)
            self.truck_artist.set_transform(rot + base_transform)

        # update text
        if self.yaw_text is not None:
            if np.isnan(deg):
                self.yaw_text.set_text("Yaw: --°")
            else:
                self.yaw_text.set_text(f"Yaw: {deg:.1f}°")

    def end(self, interrupted=False):
        self._running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def is_finished(self) -> bool:
        return not self._running
