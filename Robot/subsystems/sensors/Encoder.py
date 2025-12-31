#!/usr/bin/env python3

import time
import threading
from datetime import datetime
import logging

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
    
    
    
    def start(self, pin: int, active_high: bool = True, pull_up: bool = True, debounce_ms: int = 50,
                 edge: str = 'both'):
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

        self._running = False
        self._lock = threading.Lock()
        self._last_event_time = 0.0
        
        self.run()

        # internal history for optional export

    def _normalize_present(self, raw_state: int) -> bool:
        # raw_state is 0 or 1
        return raw_state == (1 if self.active_high else 0)

    def _gpio_callback(self, channel):
        # Keep callback extremely small: increment count only.
        with self._lock:
            self._count += 1

    def run(self):
        """Start monitoring the GPIO pin"""
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

    def update(self):
        """Update internal state (if any)"""
        pass

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

