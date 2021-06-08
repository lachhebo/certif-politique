import pandas as pd

from app.data_engineering.utils import remove_unused_columns


class DataCleaning:
    def __init__(self, features_columns, label):
        self.features_columns = features_columns
        self.label = label
        self.cie_by_avion = None

    def cleaning(self, df):
        df = df.dropna(subset=self.features_columns)
        if self.label in df.columns:
            df = df.dropna(subset=[self.label])
        return df

    def fill_na(self, df):
        if self.cie_by_avion is None:
            self.cie_by_avion = df[['CODE AVION', 'COMPAGNIE AERIENNE']].dropna().groupby(
                by=['CODE AVION']).first().to_dict()['COMPAGNIE AERIENNE']

        df.loc[df['COMPAGNIE AERIENNE'].isna(), 'COMPAGNIE AERIENNE'] = df.loc[
            df['COMPAGNIE AERIENNE'].isna(), 'CODE AVION'].apply(
                lambda x: self.cie_by_avion.get(x, "UKN")
            )
        return df

    def transform(self, df):
        df = df.copy()
        df = self.cleaning(df)
        df = remove_unused_columns(df)
        df.loc[:, "DATE"] = pd.to_datetime(df["DATE"])
        return df
