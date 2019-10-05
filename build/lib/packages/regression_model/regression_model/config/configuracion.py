import pathlib
#from os.path import dirname
import pickle

_version = '0.1.0'

PATH_CONFIG = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = pathlib.Path(PATH_CONFIG).resolve().parent

LOG_FILE = PACKAGE_ROOT / 'log_file.log'

TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
FEATURES = ['Survived','Pclass','Age','Fare', 'Survived']
TARGET = 'Survived'

features_impute_na = ['Survived','Pclass','Age','Fare']
variables_to_drop = 'Survived'
