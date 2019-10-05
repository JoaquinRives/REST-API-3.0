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


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
