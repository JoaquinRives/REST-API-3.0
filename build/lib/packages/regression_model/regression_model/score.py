import pandas as pd
import regression_model.data_management as data_management


def init():

    global model
    # retrieve the path to the model file using the model name
    pipeline_file_name = 'model.pkl'
    model = data_management.load_pipeline(file_name=pipeline_file_name)

    #model_path = Model.get_model_path('finalmodel2.pkl')
    #model = joblib.load(model_path)


def run(data):
    data = pd.read_json(data, typ='frame', orient='records')

    # make prediction
    pred = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return pred




