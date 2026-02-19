import logging
import spidev
from Robot.Constants import Constants

logger = logging.getLogger(f"{__name__}.DriveTrain")
logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output

class DAC:
    def __init__(self):
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(Constants.spi_bus, Constants.dac_spi_device)
            self._spi.max_speed_hz = Constants.dac_max_freq_hz
            self._spi.mode = Constants.dac_spi_mode
            self._resolution = Constants.dac_resolution
            self._max_value = Constants.dac_max_value

            logger.info(f"DAC SPI initialized on bus {Constants.spi_bus}, device {Constants.dac_spi_device}, mode {Constants.dac_spi_mode}")
        except Exception as e:
            logger.error(f"Failed to initialize SPI for DAC: {e}")
            self._spi = None  # Set to None to allow no-op in _write_dac
        
    def write_dac(self, channel, input):
        """Write a voltage to the specified DAC channel.

        - `channel`: integer, 0 or 1 selecting the DAC output.
        - `input`: float 0.0..1.0

        The function converts the input into 12-bit DAC counts and then
        sends the appropriate command via SPI. This is hardware-specific.
        If `spi` is None (missing spidev) this function will simply return
        silently so the code can run in environments without the DAC.
        """
        
        if self._spi is None:
            # No-op when SPI/DAC is not available (useful for debugging on PC)
            return

        # Convert input to 12-bit value (0..4095)
        value = int(input * self._max_value)
        value = max(0, min(self._max_value, value))

        # Construct command word (chip-specific) — keep existing mapping
        if channel == 0:
            command = 0x3000 | value
        else:
            command = 0xB000 | value

        # Send two bytes (MSB first) to the DAC
        self._spi.xfer2([command >> 8, command & 0xFF])
    
    def close(self):
        if self._spi:
            try:
                self._spi.close()
            except Exception:
                pass