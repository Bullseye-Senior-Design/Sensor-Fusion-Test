import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
import time
import threading
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from structure.Subsystem import Subsystem


class MPCNavigator(Subsystem):
    """Model Predictive Control Navigator for path following.
    
    This class wraps the MPC solver and provides a threaded interface for
    continuous path following using feedback from the KalmanStateEstimator.
    """
    _instance = None
    
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize MPC Navigator with default parameters."""
        # ────────────────────────────────────────────────
        # Parameters & Constants
        # ────────────────────────────────────────────────
        self.Ts = 0.1
        self.p = 12
        self.L = 0.24765
        self.v_nom = 1.0
        self.ds = self.v_nom * self.Ts
        
        # Weights (Q for state, R for input, Rd for rate of change)
        self.Q_diag = np.array([10.0, 10.0, 1.0])
        self.R_diag = np.array([0.1, 0.1])
        self.Rd_diag = np.array([1.0, 5.0])
        
        # Constraints
        self.v_bounds = [-1.0, 1.0]
        self.delta_bounds = [-np.deg2rad(30), np.deg2rad(30)]
        
        # State bounds
        self.lbx = np.array([-np.inf, -np.inf, -np.inf] * (self.p + 1) + 
                           [self.v_bounds[0], self.delta_bounds[0]] * self.p)
        self.ubx = np.array([np.inf, np.inf, np.inf] * (self.p + 1) + 
                           [self.v_bounds[1], self.delta_bounds[1]] * self.p)
        
        # Setup MPC solver
        self.solver, self.n_states, self.n_controls = self._setup_mpc()r
        
        # Constraint bounds (g=0 for initial state and dynamics)
        self.lbg = np.zeros((self.p + 1) * self.n_states)
        self.ubg = np.zeros((self.p + 1) * self.n_states)
        
        # Path matrix
        self.path_matrix = None
        
        # Thread control
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        
        # Current commands (output)
        self._v_cmd = 0.0
        self._delta_cmd = 0.0
        
        # MPC state
        self._last_u = np.array([0.0, 0.0])
        self._x_prev = None  # For warm starting
        
        # Get reference to state estimator
        self.state_estimator = KalmanStateEstimator()
    
    def _setup_mpc(self):
        """Setup the MPC solver using CasADi."""
        # Symbolic states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.size1()
        
        # Symbolic inputs
        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta)
        n_controls = controls.size1()
        
        # Right hand side (Bicycle Model)
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v/self.L * ca.tan(delta))
        f = ca.Function('f', [states, controls], [rhs])
        
        # Optimization variables
        U = ca.SX.sym('U', n_controls, self.p)
        X = ca.SX.sym('X', n_states, self.p + 1)
        P = ca.SX.sym('P', n_states + n_controls + (self.p+1)*3)
        
        cost_fn = 0
        g = []
        
        # Unpack parameters
        x_init = P[0:3]
        u_prev = P[3:5]
        ref_traj = ca.reshape(P[5:], 3, self.p+1)
        
        # Initial state constraint
        g.append(X[:, 0] - x_init)
        
        for k in range(self.p):
            st = X[:, k]
            con = U[:, k]
            
            # Tracking cost
            cost_fn += ca.mtimes([(st - ref_traj[:, k]).T, np.diag(self.Q_diag), 
                                  (st - ref_traj[:, k])])
            # Input effort cost
            cost_fn += ca.mtimes([con.T, np.diag(self.R_diag), con])
            # Smoothness cost
            u_compare = u_prev if k == 0 else U[:, k-1]
            cost_fn += ca.mtimes([(con - u_compare).T, np.diag(self.Rd_diag), 
                                  (con - u_compare)])
            
            # Dynamics constraint
            st_next = X[:, k+1]
            f_value = f(st, con)
            st_next_euler = st + (self.Ts * f_value)
            g.append(st_next - st_next_euler)
        
        # Terminal cost
        cost_fn += ca.mtimes([(X[:, self.p] - ref_traj[:, self.p]).T, 
                             np.diag(self.Q_diag), (X[:, self.p] - ref_traj[:, self.p])])
        
        # Reshape for solver
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        nlp_prob = {
            'f': cost_fn,
            'x': opt_vars,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.max_iter': 100
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts), n_states, n_controls
    
    def _generate_reference(self, cur_state):
        """Generate reference trajectory from path matrix."""
        if self.path_matrix is None:
            return np.zeros((self.p + 1, 3))
        
        x_wp = self.path_matrix[:, 0]
        y_wp = self.path_matrix[:, 1]
        theta_wp = self.path_matrix[:, 2]
        
        dx, dy = np.diff(x_wp), np.diff(y_wp)
        s_wp = np.cumsum(np.sqrt(dx**2 + dy**2))
        s_wp = np.insert(s_wp, 0, 0.0)
        
        interp_x = interp1d(s_wp, x_wp, kind='cubic', fill_value='extrapolate')
        interp_y = interp1d(s_wp, y_wp, kind='cubic', fill_value='extrapolate')
        interp_theta = interp1d(s_wp, theta_wp, kind='cubic', fill_value='extrapolate')
        
        distances = np.sqrt((x_wp - cur_state[0])**2 + (y_wp - cur_state[1])**2)
        s_cur = s_wp[np.argmin(distances)]
        
        ref = np.zeros((self.p + 1, 3))
        for i in range(self.p + 1):
            s_f = min(s_cur + i * self.ds, s_wp[-1])
            ref[i, :] = [interp_x(s_f), interp_y(s_f), interp_theta(s_f)]
        return ref
    
    def set_path(self, path_matrix):
        """Set the path to follow.
        
        Args:
            path_matrix: Nx3 array of [x, y, theta] waypoints
        """
        with self._lock:
            self.path_matrix = np.asarray(path_matrix, dtype=float)
    
    def start_path_following(self):
        """Start the MPC path following in a separate thread."""
        with self._lock:
            if self._running:
                print("Path following already running")
                return
            
            if self.path_matrix is None:
                print("Error: No path set. Call set_path() first.")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()
            print("MPC path following started")
    
    def stop_path_following(self):
        """Stop the MPC path following."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            print("MPC path following stopped")
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def get_current_commands(self):
        """Get the current velocity and steering commands.
        
        Returns:
            tuple: (velocity [m/s], steering_angle [rad])
        """
        with self._lock:
            return self._v_cmd, self._delta_cmd
    
    def is_running(self):
        """Check if path following is active."""
        with self._lock:
            return self._running
    
    def _control_loop(self):
        """Main control loop running in separate thread."""
        print("MPC control loop started")
        next_time = time.time()
        
        while True:
            with self._lock:
                if not self._running:
                    break
            
            next_time += self.Ts
            start_time = time.time()
            
            try:
                # Get current state from Kalman filter
                state = self.state_estimator.get_state()
                cur_state = np.array([state.pos[0], state.pos[1], 
                                     self.state_estimator.euler[2]])  # x, y, yaw
                
                # Generate reference trajectory
                refs = self._generate_reference(cur_state)
                
                # Prepare parameters
                params = np.concatenate([cur_state, self._last_u, refs.flatten()])
                
                # Solve MPC
                solver_args = {
                    'lbx': self.lbx, 
                    'ubx': self.ubx, 
                    'lbg': self.lbg, 
                    'ubg': self.ubg, 
                    'p': params
                }
                if self._x_prev is not None:
                    solver_args['x0'] = ca.DM(self._x_prev)
                
                res = self.solver(**solver_args)
                
                # Extract control commands
                u_offset = self.n_states * (self.p + 1)
                u_opt = res['x'][u_offset : u_offset + self.n_controls]
                v_cmd = float(u_opt[0])
                delta_cmd = float(u_opt[1])
                
                # Update stored values
                with self._lock:
                    self._v_cmd = v_cmd
                    self._delta_cmd = delta_cmd
                    self._last_u = np.array([v_cmd, delta_cmd])
                    self._x_prev = res['x']
                
                elapsed = time.time() - start_time
                print(f"MPC: V={v_cmd:.2f} m/s | δ={np.degrees(delta_cmd):.1f}° | "
                      f"Pos=({cur_state[0]:.2f}, {cur_state[1]:.2f}) | Time={elapsed:.3f}s")
                
            except Exception as e:
                print(f"MPC error: {e}")
                with self._lock:
                    self._v_cmd = 0.0
                    self._delta_cmd = 0.0
            
            # Timing control
            sleep_duration = next_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        
        print("MPC control loop stopped")


# ────────────────────────────────────────────────
# Example Usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Create navigator
    navigator = MPCNavigator()
    
    # Create a simple straight path
    num_points = 200
    path_matrix = np.zeros((num_points, 3))
    path_matrix[:, 0] = np.linspace(0, 100, num_points)  # Straight line in x
    
    # Set path and start
    navigator.set_path(path_matrix)
    navigator.start_path_following()
    
    try:
        # Poll for commands
        while True:
            time.sleep(0.5)
            v, delta = navigator.get_current_commands()
            print(f"Current commands: V={v:.2f} m/s, δ={np.degrees(delta):.1f}°")
    except KeyboardInterrupt:
        print("\nStopping...")
        navigator.stop_path_following()