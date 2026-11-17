"""One simple way to create loggers for the whole project.

Why logging and not print()?
- Each line gets a time and a level (INFO, WARNING, ERROR).
- Lines are also saved to a file, so we can read them later.
  This is very important when a pipeline runs on a server.
"""

import logging

from src.sms_fraud.config import get_config, project_path

# How each log line looks, for example:
# 2026-11-17 10:15:02 | INFO | src.sms_fraud.data | Loaded 5574 rows
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to logs/app.log."""
    logger = logging.getLogger(name)

    # Python keeps one logger per name. If we already added handlers to it,
    # we must not add them again, or every line would be printed twice.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Handler 1: show logs in the console.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2: save logs to a file. Create the logs/ folder if needed.
    log_file = project_path(get_config()["logging"]["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Do not also send lines to the "root" logger (avoids double printing).
    logger.propagate = False
    return logger
