import logging
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Creates a centralized logger that outputs to both the console and a file.
    """
    logger = logging.getLogger(name)
    
    # Only configure if the logger doesn't already have handlers to prevent duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Standard production format: [Time] [Module] [Level] - Message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 1. Console Handler (for terminal visibility)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (for persistent tracking)
        file_handler = logging.FileHandler(settings.LOG_FILE_PATH, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger