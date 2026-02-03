

class Clutches:
    
    _instance = None
    
    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def start(self, left_clutch_pin: int, right_clutch_pin: int):
        """Initialize clutch control GPIO pins and state."""
        # GPIO pin setup code here
        self.left_clutch_pin = left_clutch_pin
        self.right_clutch_pin = right_clutch_pin
        
        pass
    
    def engage_left_clutch(self):
        """Engage the left clutch."""
        pass