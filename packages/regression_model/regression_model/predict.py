import pandas as pd

from regression_model.config import configuracion
import regression_model.data_management as data_management
import logging


_logger = logging.getLogger(__name__)

pipeline_file_name = 'model.pkl'


def make_prediction(json_data):
    """Make a prediction using the saved model pipeline."""

    model = data_management.load_pipeline(file_name=pipeline_file_name)

    input_data = pd.read_json(json_data)
    prediction = model.predict(input_data)

    _logger.info(
        f'Making predictions with model version: {configuracion._version} '
        f'Inputs: {json_data} '
        f'Predictions: {str(prediction)}')

    return prediction

