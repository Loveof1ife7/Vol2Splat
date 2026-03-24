import logging
import sys
from typing import Optional

# Create a custom logger
logger = logging.getLogger("vol2splat")
logger.setLevel(logging.INFO)

# Create handlers
# Console handler
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(c_handler)

def setup_logging(level: str = "INFO"):
    """Configure logging level."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)
    c_handler.setLevel(numeric_level)

def get_logger(name: Optional[str] = None):
    """Get a child logger."""
    if name:
        return logger.getChild(name)
    return logger
