from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from regression_model import preprocessors as pp
from regression_model.config import configuracion

preprocessor_pipe = Pipeline(
    [
        ('random_imputer',
            pp.ImputeNa(variables=configuracion.features_impute_na)),
        ('DropFeatures',
            pp.DropUnnecessaryFeatures(variables_to_drop=configuracion.variables_to_drop)),
        ('scaler', StandardScaler()),
        ('L_model', RandomForestClassifier())
    ]
)



