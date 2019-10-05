from azureml.core import Workspace
from azureml.core import Experiment
from regression_model.data_management import load_dataset
from regression_model.config import configuracion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ws = Workspace.from_config()

experiment = Experiment(workspace=ws, name="finalexp2")

data = load_dataset(file_name=configuracion.TRAINING_DATA_FILE)
y = data[configuracion.TARGET]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=66)

#from data_management import load_pipeline
#import configuracion
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class ImputeNa(BaseEstimator, TransformerMixin):
    """ Replace nan values with 'missing' """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        for variable in self.variables:
            self.mean_dict[variable] = X[variable].mean()

        return self

    def transform(self, X):
        for variable in self.variables:
            X[variable] = X[variable].fillna(self.mean_dict[variable])
        return X

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X

preprocessor_pipe = Pipeline(
    [
        ('random_imputer',
            ImputeNa(variables=['Survived','Pclass','Age','Fare'])),
        ('DropFeatures',
            DropUnecessaryFeatures(variables_to_drop='Survived')),
        ('scaler', StandardScaler()),
        ('L_model', RandomForestClassifier())
    ]
)


run = experiment.start_logging()

model = preprocessor_pipe
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
run.log("acc", accuracy)

model_name = "finalmodel2.pkl"
filename = "finalmodel2.pkl"

joblib.dump(value=model, filename='finalmodel2.pkl')
run.upload_file(name=model_name, path_or_stream=filename)
run.complete()


from azureml.core import Run

runid='062ae84f-1f94-43ed-bcf5-377a93b14006'
run = Run(experiment=experiment, run_id=runid)
print(run.get_file_names())


# Change names
run.download_file(name="finalmodel2.pkl")

model = run.register_model(model_name='finalmodel2.pkl',
                           model_path='finalmodel2.pkl')
print(model.name, model.id, model.version, sep='\t')













#############################################################
n_estimators = [7, 8, 9, 10, 11]

for n in n_estimators:
    run = experiment.start_logging()
    run.log("n_estimators", n)

    model = preprocessor_pipe.set_params(L_model__n_estimators=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    run.log("acc", accuracy)

    model_name = "model_n_estimators_" + str(n) + ".pkl"
    filename = "outputs/" + model_name

    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()

    ###############

maximum_acc_runid = None
maximum_acc = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
    # each logged metric becomes a key in this returned dict
    run_acc = run_metrics["acc"]
    run_id = run_details["runId"]

    if maximum_acc is None:
        maximum_acc = run_acc
        maximum_acc_runid = run_id
    else:
        if run_acc > maximum_acc:
            maximum_acc = run_acc
            maximum_acc_runid = run_id

print("Best run_id: " + maximum_acc_runid)
print("Best run_id acc: " + str(maximum_acc))

from azureml.core import Run
best_run = Run(experiment=experiment, run_id=maximum_acc_runid)
print(best_run.get_file_names())

################################
# Change names
best_run.download_file(name="model_n_estimators_8.pkl")

model = best_run.register_model(model_name='model_2',
                           model_path='model_n_estimators_8.pkl')
print(model.name, model.id, model.version, sep='\t')
################################################################

