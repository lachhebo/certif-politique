import pandas as pd
from sklearn import preprocessing


class MultiColumnLabelEncoder:
    def __init__(self, encoded_columns=None, all_columns=False):
        if encoded_columns is None:
            self.encoded_columns = []
        else:
            self.encoded_columns = encoded_columns  # array of column names to encode
        self.label_encoder = {}
        self.all_columns = all_columns

    def fit(self, x: pd.DataFrame):
        """
        Fit columns of x specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in x.
        """
        encoded_columns = list(x.columns) if self.all_columns else self.encoded_columns
        for column_name in encoded_columns:
            self.label_encoder[column_name] = preprocessing.LabelEncoder().fit(
                x[column_name]
            )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in X.

        Values not seen in fit phase are not handle
        """
        output = x.copy()

        encoded_columns = list(x.columns) if self.all_columns else self.encoded_columns

        for column_name in encoded_columns:
            le_dict = dict(
                zip(
                    self.label_encoder[column_name].classes_,
                    self.label_encoder[column_name].transform(
                        self.label_encoder[column_name].classes_
                    ),
                )
            )
            output[column_name] = output[column_name].apply(
                lambda y: le_dict.get(y, -1)
            )

        return output

    def fit_transform(self, x):
        return self.fit(x).transform(x)
