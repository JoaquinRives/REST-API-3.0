import logging
from regression_model.config import configuracion
from logging.handlers import TimedRotatingFileHandler
import os
import sys

from regression_model.config import configuracion

# Multiple calls to logging.getLogger('someLogger') return a
# reference to the same logger object.  This is true not only
# within the same module, but also across modules as long as
# it is in the same Python interpreter process.

FORMAT = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")


# def get_console_handler():
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(FORMATTER)
#     return console_handler


def set_logger(logger):
    # Config level
    logging.basicConfig(
        level=logging.DEBUG)  # To log everything, by default it only logs warning and above.

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(configuracion.LOG_FILE)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_handler.setFormatter(FORMAT)
    f_handler.setFormatter(FORMAT)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.propagate = False

    return logger
