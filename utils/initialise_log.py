import logging
import os
from pathlib import Path
import logging.config


def getLogger(log_fileName: str):
    # Load logging configuration from lo_config.ini
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(base_dir).parent
    log_fileName = f"{parent_dir}/config/{log_fileName}"
    logging.config.fileConfig(log_fileName, disable_existing_loggers=False)
    # Use logging as usual
    logger = logging.getLogger(__name__)
    return logger
