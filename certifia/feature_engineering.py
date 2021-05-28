from certifia.utils.multi_column_label_encode import MultiColumnLabelEncoder
import pandas as pd


class FeatureEngineering:
    def __init__(self, training_columns=None, columns_to_dummify=None):
        self.training_columns = training_columns
        self.columns_to_dummify = columns_to_dummify
        self.label_encoder = MultiColumnLabelEncoder(columns=self.columns_to_dummify)

    def cleaning(self, df):
        df = df.dropna(subset=["RETARD A L'ARRIVEE"])
        # NIVEAU DE SECURITE is always the same value
        df = df.drop(columns=['NIVEAU DE SECURITE'])
        return df

    def split_feature_label(self, df):
        X = df[self.training_columns]
        y = df["RETARD A L'ARRIVEE"]
        return X, y

    def fit_transform_dummify_columns(self, X):
        if self.columns_to_dummify is not None:
            return self.label_encoder.fit_transform(X)
        return X

    def transform_dummify_columns(self, X):
        if self.columns_to_dummify is not None:
            return self.label_encoder.transform(X)
        return X

    def fit(self, dataframe: pd.DataFrame):
        df = dataframe.copy()

        df = self.cleaning(df)

        X, y = self.split_feature_label(df)

        X = self.fit_transform_dummify_columns(X)

        return X, y

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy()
        df = df[self.training_columns]
        return self.transform_dummify_columns(df)
