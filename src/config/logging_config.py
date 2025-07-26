import logging.config
import os

def setup_logging():
    """
    Sets up logging using a dictionary configuration.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "file_formatter": {
                "format": "[%(asctime)s] %(module)s - %(levelname)s - %(message)s",
            },
            "console_formatter": {
                "format": "%(filename)s:%(lineno)d:%(funcName)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",  
                "formatter": "console_formatter",
                "stream": "ext://sys.stdout",  
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",  
                "formatter": "file_formatter",
                "filename": os.path.join(log_dir, "events.log"),
                "maxBytes": 100 * 1024,  # 100 KB
                "backupCount": 3,
                "encoding": "utf8",
            },
        },
        "root": {
            "level": "INFO",  
            "handlers": ["console", "file"]
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)