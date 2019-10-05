from regression_model.config import configuracion
from regression_model import data_management
from regression_model import preprocessors
from regression_model import pipeline
from regression_model import train_pipeline
from regression_model import predict
from regression_model.config import logging_config


import logging
import os

from regression_model import configuracion
from regression_model.config import logging_config

# Configure logger for use in package
logger = logging.getLogger(__name__)

logger = logging_config.set_logger(logger)
