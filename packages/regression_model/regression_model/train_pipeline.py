import regression_model.pipeline as pipeline
import regression_model.data_management as data_management
from regression_model.config import configuracion

import logging


# Logging
_logger = logging.getLogger(__name__)


def run_training():
    """Train the model."""

    # Get training data
    data = data_management.load_dataset(file_name=configuracion.TRAINING_DATA_FILE)
    y = data[configuracion.TARGET]

    model = pipeline.preprocessor_pipe.fit(data, y)

    data_management.save_pipeline(model_to_persist=model)

    _logger.debug(f'Training model version: {configuracion._version} ')


if __name__ == '__main__':
    run_training()

