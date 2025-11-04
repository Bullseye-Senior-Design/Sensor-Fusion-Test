from structure.commands.Command import Command

import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib import patches
from matplotlib import transforms as mtransforms

from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.sensors.IMU import IMU


class PlotStateCmd(Command):
    """Command that plots the EKF position (x,y) in a Matplotlib figure embedded
    in a non-blocking Tkinter window.

    Usage: create and schedule this command from your RobotContainer or a script.
    The command will keep the window open until the command is cancelled or the
    user closes the window.
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
        self.truck_patch = None
        self.yaw_text = None

        # estimator instance
        self.estimator = KalmanStateEstimator()
        self.imu = IMU()

    def initalize(self):
        # Create Tk window and Matplotlib canvas. We do NOT call mainloop;
        # instead we call root.update() from execute() so the window is non-blocking.
        if tk is None:
            print("PlotStateCmd: tkinter not available; cannot create GUI.")
            self._running = False
            return

        try:
            self.root = tk.Tk()
        except Exception as e:  # pragma: no cover - depends on environment
            print(f"PlotStateCmd: failed to create Tk root: {e}")
            self.root = None
            self._running = False
            return

        self.root.wm_title("EKF Position Plot")

        # Use two subplots: main XY plot on left and a small top-down yaw view
        self.figure = Figure(figsize=(8, 4), dpi=100)
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        # main XY plot
        self.ax = self.figure.add_subplot(gs[0, 0])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        # initial empty line
        self.line, = self.ax.plot([], [], "b.-", markersize=4)

        # top-down yaw view on right
        self.ax_top = self.figure.add_subplot(gs[0, 1])
        self.ax_top.set_title("Top-down (yaw)")
        self.ax_top.set_xlim(-1.5, 1.5)
        self.ax_top.set_ylim(-1.5, 1.5)
        self.ax_top.set_aspect("equal")
        self.ax_top.axis("off")

        # draw a simple truck shape as a rectangle + cabin triangle centered at origin
        truck_length = 1.0
        truck_width = 0.6
        # rectangle centered at origin (lower-left at -L/2, -W/2)
        rect = patches.Rectangle(
            (-truck_length / 2.0, -truck_width / 2.0),
            truck_length,
            truck_width,
            facecolor="#d4a017",
            edgecolor="k",
            linewidth=1.0,
        )
        # small cabin triangle at front
        cabin = patches.Polygon(
            [
                (truck_length / 2.0, 0.0),
                (truck_length / 2.0 - 0.15, truck_width / 4.0),
                (truck_length / 2.0 - 0.15, -truck_width / 4.0),
            ],
            closed=True,
            facecolor="#c77b00",
            edgecolor="k",
        )

        # add to axis and keep references for rotation updates
        self.truck_patch = rect
        self.ax_top.add_patch(rect)
        self.ax_top.add_patch(cabin)
        # text below the truck to show numeric yaw
        self.yaw_text = self.ax_top.text(0.0, -1.2, "Yaw: --\N{DEGREE SIGN}", ha="center", va="center")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # keep running until closed
        self._running = True

    def execute(self):
        # If GUI couldn't be created, nothing to do
        if not self._running or self.root is None:
            return

        # get current position from EKF
        try:
            pos = self.estimator.pos  # numpy array [x,y,z]
        except Exception as e:
            # If estimator fails for any reason, just skip this update
            print(f"PlotStateCmd: failed to read estimator: {e}")
            return

        # safe conversion (in case of None or invalid values)
        try:
            x = float(pos[0])
        except Exception:
            x = np.nan
        try:
            y = float(pos[1])
        except Exception:
            y = np.nan

        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > self.max_points:
            self.xs = self.xs[-self.max_points :]
            self.ys = self.ys[-self.max_points :]

        # update line data and autoscale
        self.line.set_data(self.xs, self.ys) # type: ignore
        self.ax.relim() # type: ignore
        self.ax.autoscale_view() # type: ignore

        # draw and process Tk events in a non-blocking way
        try:
            self.canvas.draw_idle() # type: ignore
            # If draw_idle doesn't immediately draw in some backends, force draw
            self.canvas.draw() # type: ignore
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            # If window was closed by user, mark command finished
            self._running = False

        # update top-down yaw view (do this after drawing to avoid flicker)
        if self.ax_top is not None:
            euler = self.imu.get_euler()  # [roll, pitch, yaw]
            yaw = float(euler[2])


            # apply rotation to truck patches around origin
            trans = mtransforms.Affine2D().rotate(yaw) + self.ax_top.transData
            # set same transform for all patches in axis
            for p in list(self.ax_top.patches):
                p.set_transform(trans)

            # update yaw text in degrees
            try:
                deg = np.degrees(yaw)
                if self.yaw_text is not None:
                    self.yaw_text.set_text(f"Yaw: {deg:.1f}\N{DEGREE SIGN}")
            except Exception:
                if self.yaw_text is not None:
                    self.yaw_text.set_text("Yaw: --\N{DEGREE SIGN}")

    def end(self, interrupted):
        # close window and cleanup
        self._running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def is_finished(self) -> bool:
        # the command finishes when the window is closed or end() is called
        return not self._running
