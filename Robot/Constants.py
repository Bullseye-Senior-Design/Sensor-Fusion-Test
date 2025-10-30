class Constants:
    # UWB Tag Data
    class UWBTagInfo:
        def __init__(self, port, offset):
            self.port = port
            self.offset = offset

    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyUSB0", offset=(0.1, 0.0, 0.0)),
                    UWBTagInfo(port="/dev/ttyUSB1", offset=(0.2, 0.0, 0.0)) ]