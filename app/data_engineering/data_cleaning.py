import pandas as pd


class DataCleaning:
    def __init__(self, features_columns, label):
        self.features_columns = features_columns
        self.label = label

    def remove_unused_columns(self, df):
        if 'NIVEAU DE SECURITE' in df.columns:
            df = df.drop(columns=['NIVEAU DE SECURITE'])
        return df

    def cleaning(self, df):
        df = df.dropna(subset=self.features_columns)
        if self.label in df.columns:
            df = df.dropna(subset=[self.label])
        return df

    def transform(self, df):
        df = df.copy()
        df = self.cleaning(df)
        df = self.remove_unused_columns(df)
        df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'])
        return df
