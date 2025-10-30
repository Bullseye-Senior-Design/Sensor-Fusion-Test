class UWBTagInfo:
    def __init__(self, port, offset):
        self.port = port
        self.offset = offset

class Constants:
    # UWB Tag Data
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyUSB0", offset=(-13.335 / 2, -22.86 / 2, 0.0)),
                    UWBTagInfo(port="/dev/ttyUSB1", offset=(13.335 / 2, 22.86 / 2, 0.0)) ]
