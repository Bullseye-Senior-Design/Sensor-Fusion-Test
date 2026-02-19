import logging
from typing import Optional
import spidev
from Robot.Constants import Constants

logger = logging.getLogger(f"{__name__}.DriveTrain")
logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output

class FrontWheelEncoder:
    def __init__(self):
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(Constants.spi_bus, Constants.frontwheel_encoder_spi_device)
            self._spi.max_speed_hz = Constants.frontwheel_encoder_max_freq_hz
            self._spi.mode = Constants.frontwheel_encoder_spi_mode
            self._resolution = Constants.frontwheel_encoder_resolution
            self._max_position = Constants.frontwheel_encoder_max_position

            logger.info(f"FrontWheelEncoder SPI initialized on bus {Constants.spi_bus}, device {Constants.frontwheel_encoder_spi_device}, mode {Constants.frontwheel_encoder_spi_mode}")
        except Exception as e:
            logger.error(f"Failed to initialize SPI for FrontWheelEncoder: {e}")
            self._spi = None  # Set to None to allow no-op in _read_raw_position

    def read_position(self) -> Optional[int]:
        """Read raw position data from encoder via SPI.
        
        For SE33SPI encoders, the typical protocol is:
        - Send a read command (or dummy bytes)
        - Receive position data (typically 2 bytes for 12-bit resolution)
        
        Returns:
            Raw position value (0 to max_position) or None on error
        """
        if not self._spi:
            return None

        try:
            # For most SPI absolute encoders:
            # Send 2-3 bytes of 0x00 to clock out the position data
            # Adjust based on your specific encoder protocol
            bytes_to_read = (self._resolution + 7) // 8  # Round up to nearest byte
            if bytes_to_read < 2:
                bytes_to_read = 2
            
            # Read data from encoder
            data = self._spi.readbytes(bytes_to_read)
            
            # Parse position based on resolution
            # For 12-bit encoder in 2 bytes: MSB first
            # Example: [0x1F, 0xA3] -> 0x1FA3 (12 bits used)
            position = 0
            for byte_val in data:
                position = (position << 8) | byte_val
            
            # Mask to resolution bits
            position &= self._max_position
            
            return position

        except Exception as e:
            logger.error(f"FrontWheelEncoder SPI read error: {e}")
            return None
        
    def close(self):
        if self._spi:
            try:
                self._spi.close()
            except Exception:
                pass