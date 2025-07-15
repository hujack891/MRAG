import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BaseConfig
import selenium
import urllib3

baseconfig = BaseConfig()


def setup_logging(file_name: str):
    log_dir = baseconfig.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log file path
    log_file = os.path.join(log_dir, f"{file_name}.log")

    # If the file already exists, delete the old log file first.
    if os.path.exists(log_file):
        os.remove(log_file)

    if baseconfig.LOGGING_LEVEL == 'DEBUG':
        # Configure the logger
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),  
                logging.StreamHandler()  

            ]
        )
    elif baseconfig.LOGGING_LEVEL == 'INFO':
        # Configure the logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"), 
                logging.StreamHandler()  

            ]
        )


    logging.getLogger('selenium.webdriver.remote.remote_connection').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    
    logger = logging.getLogger(file_name)
    if baseconfig.LOGGING_LEVEL == 'DEBUG':
        logger.critical('Currently in debug mode')
    elif baseconfig.LOGGING_LEVEL == 'INFO':
        logger.critical('Currently in application mode')
    return logger
