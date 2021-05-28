from sklearn import preprocessing


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.label_encoder = {}

    def fit(self, X, y=None):
        """
        Fit columns of X specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in X.
        """
        if type(self.columns) is list:
            for col in self.columns:
                self.label_encoder[col] = preprocessing.LabelEncoder().fit(X[col])
        elif type(self.columns) is str and self.columns == 'all':
            for colname, col in X.iteritems():
                self.label_encoder[colname] = preprocessing.LabelEncoder().fit(col)
        return self

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If 'all' is given, transforms all
        columns in X.

        Values not seen in fit phase are not handle
        """
        output = X.copy()
        if type(self.columns) is list:
            for col in self.columns:
                output[col] = self.label_encoder[col].transform(output[col])
        elif type(self.columns) is str and self.columns == 'all':
            for colname, col in output.iteritems():
                output[colname] = self.label_encoder[colname].transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
