class UWBTagInfo:
    def __init__(self, port, offset):
        self.port = port
        self.offset = offset

class Constants:
    # UWB Tag Data
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyACM0", offset=(-13.335 / 2 / 100, -22.86 / 2 / 100, 0.0)),
                    UWBTagInfo(port="/dev/ttyACM1", offset=(13.335 / 2 / 100, 22.86 / 2 / 100, 0.0)) ]
    
    # Encoder Constants
    back_right_encoder_pin = 7
