class UWBTagInfo:
    def __init__(self, port, offset):
        self.port = port
        self.offset = offset

class Constants:
    # UWB Tag Data
    # For the tag offsets:
    # +x : tag is forward of the robot center
    # +y : tag is to the robot's left side
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyACM1", offset=(-22.86 / 2 / 100, 13.335 / 2 / 100, 0.0)), # back left tag
                    UWBTagInfo(port="/dev/ttyACM0", offset=(22.86 / 2 / 100, -13.335 / 2 / 100, 0.0)) ] # front right tag

