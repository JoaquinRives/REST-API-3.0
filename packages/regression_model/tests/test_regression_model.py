from regression_model.data_management import load_dataset
from regression_model import predict
from regression_model import train_pipeline


# Testing train_pipeline.py
def test_training():
    train_pipeline.run_training()


# Testing predict (single input):
def test_make_single_prediction():
    test_data = load_dataset(file_name='test.csv')
    single_test_input = test_data[0:1].to_json(orient='records')

    subject = predict.make_prediction(single_test_input)

    assert subject is not None
    assert isinstance(subject[0], float)


# Testing predict (multiple input):
def test_multiple_prediction():
    test_data = load_dataset(file_name='test.csv')
    multiple_test_input = test_data.to_json(orient='records')

    subject = predict.make_prediction(multiple_test_input)

    assert subject is not None
    assert len(subject) == 482


# def main():
#     test_training()
#     test_make_single_prediction()
#     test_multiple_prediction()
#
#
# if __name__ == '__main__':
#     main()



