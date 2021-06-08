import pickle

import pandas as pd

from app import ROOT_PATH


class DataCleaning:
    def __init__(self, features_columns, label):
        self.features_columns = features_columns
        self.label = label
        self.cie_by_avion = None

    def fill_na(self, df):
        if self.cie_by_avion is None:
            self.cie_by_avion = df[['CODE AVION', 'COMPAGNIE AERIENNE']].dropna().groupby(
                by=['CODE AVION']).first().to_dict()['COMPAGNIE AERIENNE']

        df.loc[df['COMPAGNIE AERIENNE'].isna(), 'COMPAGNIE AERIENNE'] = df.loc[
            df['COMPAGNIE AERIENNE'].isna(), 'CODE AVION'].apply(
                lambda x: self.cie_by_avion.get(x, "UKN")
            )
        return df

    def drop_na(self, df):
        df = df.dropna(subset=self.features_columns)
        if self.label in df.columns:
            df = df.dropna(subset=[self.label])
        return df

    def fit_drop(self, df):
        df = self.fill_na(df)
        df = self.drop_na(df)
        return df

    def fit(self, df):
        df = df.copy()
        df = self.fit_drop(df)
        df = df[df['NOMBRE DE PASSAGERS'] < 1000]
        df = df[df["RETARD A L'ARRIVEE"] < 250]
        df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'])
        return df

    def transform(self, df):
        df = df.copy()
        df = self.fit_drop(df)
        df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'])
        return df

    def save_cleaner(self, path=None) -> None:
        """
        Save to file in the current working directory
        """
        if path is None:
            path = (ROOT_PATH / "data" / "output" / "data_cleaner.pkl").resolve()
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_cleaner(self, path=None):
        """
        Load to file in the current working directory
        """
        if path is None:
            path = (ROOT_PATH / "data" / "output" / "data_cleaner.pkl").resolve()
        with open(path, "rb") as file:
            pickle_data_cleaning = pickle.load(file)
            self.features_columns = pickle_data_cleaning.features_columns
            self.label = pickle_data_cleaning.label
            self.cie_by_avion = pickle_data_cleaning.cie_by_avion
        return self
