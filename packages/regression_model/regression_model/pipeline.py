from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from regression_model import preprocessors as pp
from regression_model.config import configuracion


preprocessor_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=configuracion.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            pp.NumericalImputer(variables=configuracion.NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            pp.TemporalVariableEstimator(
                variables=configuracion.TEMPORAL_VARS,
                reference_variable=configuracion.DROP_FEATURES)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=configuracion.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=configuracion.CATEGORICAL_VARS)),
        ('log_transformer',
            pp.LogTransformer(variables=configuracion.NUMERICALS_LOG_VARS)),
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=configuracion.DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('Linear_model', Lasso(alpha=0.005, random_state=0))
    ]
)



