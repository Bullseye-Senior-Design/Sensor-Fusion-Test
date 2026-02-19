class UWBTagInfo:
    def __init__(self, port, id, offset):
        self.port = port
        self.id = id
        self.offset = offset

class Constants:
    # SPI Constants
    spi_bus = 0
    
    # DAC Constants
    dac_spi_device = 0
    dac_spi_mode = 0
    dac_max_freq_hz = 5000
    dac_backwheel_channel = 0
    dac_frontwheel_channel = 1
    dac_resolution = 12
    dac_max_value = (1 << dac_resolution) - 1  # 4095 for 12-bit DAC

    # Front Wheel Encoder Constants
    frontwheel_encoder_spi_device = 1
    frontwheel_encoder_spi_mode = 0
    frontwheel_encoder_max_freq_hz = 100000
    frontwheel_encoder_resolution = 12  # bits
    frontwheel_encoder_max_position = (1 << frontwheel_encoder_resolution) - 1
    
    # Back Wheel Encoder Constants
    back_right_encoder_pin = 4
    back_left_encoder_pin = 5
    wheel_circumference = 0.25  # meters
    counts_per_revolution = 6  # encoder pulses per wheel rotation
    
    # Back Wheel Constants
    backwheel_forward_ssr_pin = 22
    backwheel_reverse_ssr_pin = 27
    backwheel_power_ssr_pin = 17
    rear_motor_top_speed = 0.13
    
    # Front Wheel Constants
    frontwheel_power_ssr_pin = 23
    
    # Clutches Constants
    left_clutch_pin = 17
    right_clutch_pin = 27
    
    # For the tag offsets:
    # +x : tag is forward of the robot center
    # +y : tag is to the robot's left side
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyACM0", id=0, offset=(-24.77 / 2 / 100, 22.225 / 2 / 100, 0.0)), # back left tag
                    UWBTagInfo(port="/dev/ttyACM1", id=1, offset=(24.77 / 2 / 100, -22.225 / 2 / 100, 0.0)) ] # front right tag

