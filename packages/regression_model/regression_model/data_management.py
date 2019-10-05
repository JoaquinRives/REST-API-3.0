
import pandas as pd
from regression_model.config import configuracion
from sklearn.externals import joblib


def load_dataset(file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{configuracion.DATASET_DIR}/{file_name}', index_col=0)
    return _data


def save_pipeline(model_to_persist):
    """To save the model pipeline"""

    save_file_name = 'model.pkl'
    save_path = configuracion.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(model_to_persist, save_path)

    print('saved pipeline')


def load_pipeline(file_name):
    """Load the model pipeline"""

    file_path = configuracion.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

