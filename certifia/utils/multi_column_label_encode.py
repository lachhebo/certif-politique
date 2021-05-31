import pandas as pd
from sklearn import preprocessing


class MultiColumnLabelEncoder:
    def __init__(self, columns=None, all_columns=False):
        self.columns = columns  # array of column names to encode
        self.label_encoder = {}
        self.all_columns = all_columns

    def fit(self, X: pd.DataFrame):
        """
        Fit columns of X specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in X.
        """
        if self.columns is not None:
            for colname in self.columns:
                self.label_encoder[colname] = preprocessing.LabelEncoder().fit(X[colname])
        elif self.all_columns:
            for colname, col in X.iteritems():
                self.label_encoder[colname] = preprocessing.LabelEncoder().fit(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in X.

        Values not seen in fit phase are not handle
        """
        output = X.copy()
        if self.columns is not None:
            for colname in self.columns:
                output[colname] = self.label_encoder[colname].transform(output[colname])
        elif self.all_columns is True:
            for colname, col in output.iteritems():
                output[colname] = self.label_encoder[colname].transform(col)
        return output

    def fit_transform(self, X):
        return self.fit(X).transform(X)
