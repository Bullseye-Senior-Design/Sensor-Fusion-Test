#!/usr/bin/env python3

import time
import threading
from datetime import datetime
import logging
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator

logger = logging.getLogger("ProximitySensor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import RPi.GPIO as GPIO

class Encoder:
    
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def start(self, pin: int, active_high: bool = True, pull_up: bool = True, debounce_ms: int = 100,
                 edge: str = 'rising', wheel_circumference: float = 0.5, counts_per_revolution: int = 6):
        """Create a proximity sensor reader.

        Args:
            pin: BCM GPIO pin number to read (e.g., 17)
            active_high: True if sensor output is HIGH when object detected
            pull_up: True to enable pull-up, False for pull-down (when using internal pull)
            debounce_ms: Debounce time in milliseconds
            edge: 'rising', 'falling', or 'both' (which edge to detect)
        """
        self.pin = pin
        self.active_high = active_high
        self.pull_up = pull_up
        self.debounce_ms = debounce_ms
        self.edge = edge.lower()

        self.interval = 1  # update interval in seconds
        self._running = False
        self._lock = threading.Lock()
        self._last_event_time = 0.0
        self._count = 0
        self._velocity = 0.0  # m/s
        self._last_update_time = time.time()
        self.state_estimator = KalmanStateEstimator()
        logger.info("edge {edge}, debounce {debounce}ms".format(edge=self.edge, debounce=self.debounce_ms))
        
        # Wheel parameters (customize these)
        self.wheel_circumference = wheel_circumference  # meters (adjust to your wheel)
        self.counts_per_revolution = counts_per_revolution  # encoder pulses per wheel rotation
        
        self.run()

        # internal history for optional export

    def _normalize_present(self, raw_state: int) -> bool:
        # raw_state is 0 or 1
        return raw_state == (1 if self.active_high else 0)

    def _gpio_callback(self, channel):
        # Keep callback extremely small: increment count only.
        logger.info(f"callback callled count={self._count}")
        with self._lock:
            self._count += 1

    def run(self):
        """Start monitoring the GPIO pin"""
        # Prevent multiple threads from being started
        if self._running:
            logger.warning(f"Encoder already running on GPIO {self.pin}")
            return
        
        self._running = True

        GPIO.setmode(GPIO.BCM)

        pud = GPIO.PUD_UP if self.pull_up else GPIO.PUD_DOWN
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=pud)

        # Determine edge type
        if self.edge == 'both':
            gedge = GPIO.BOTH
        elif self.edge == 'rising':
            gedge = GPIO.RISING
        elif self.edge == 'falling':
            gedge = GPIO.FALLING
        else:
            gedge = GPIO.BOTH

        GPIO.add_event_detect(self.pin, gedge, callback=self._gpio_callback, bouncetime=self.debounce_ms)
        logger.info(f"Started monitoring GPIO {self.pin} (active_high={self.active_high})")
        
        def _update_loop():
            while True:
                self.update()
        
        threading.Thread(target=_update_loop, daemon=True).start()

    def update(self):
        """Update velocity estimate from encoder counts"""
        current_time = time.time()
        dt = current_time - self._last_update_time
            
        if dt > 0:
            # Calculate velocity from count changes
            distance = (self.get_count_and_reset() / self.counts_per_revolution) * self.wheel_circumference
            
            logger.info(f"count ={self._count} reset for next interval")
            self._velocity = distance / dt
            logger.info(f"Encoder velocity: {self._velocity:.3f} m/s over dt={dt:.3f}s with count={self._count}")
            self.state_estimator.update_encoder_velocity(self._velocity)
                
            # Reset for next 
            self._last_update_time = current_time
        
        time.sleep(self.interval)

    def stop(self):
        """Stop monitoring and cleanup"""
        self._running = False

        if GPIO is not None:
            try:
                GPIO.remove_event_detect(self.pin)
            except Exception:
                pass
            # Don't call GPIO.cleanup() globally to avoid affecting other users; only cleanup pin
            try:
                GPIO.cleanup(self.pin)
            except Exception:
                pass

    def get_count(self) -> int:
        """Return the number of callbacks since last reset and reset the counter."""
        with self._lock:
            c = self._count
        return c
    
    def get_count_and_reset(self) -> int:
        """Return the number of callbacks since last reset and reset the counter."""
        with self._lock:
            c = self._count
            self._count = 0
        return c

    def get_velocity(self) -> float:
        """Return current velocity estimate in m/s (forward direction)"""
        return self._velocity

