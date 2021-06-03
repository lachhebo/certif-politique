import pickle
from typing import Tuple, Dict

import pandas as pd

from app.utils.multi_column_label_encode import MultiColumnLabelEncoder


class FeatureEngineering:
    def __init__(
        self,
        training_columns=None,
        columns_to_dummify=None,
        label_name="RETARD A L'ARRIVEE",
    ):
        self.training_columns = training_columns
        self.columns_to_dummify = columns_to_dummify
        self.label_encoder = MultiColumnLabelEncoder(columns=self.columns_to_dummify)
        self.label_name = label_name

    def cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaning_columns = list(set(df.columns.intersection(self.training_columns)))
        cleaning_columns.append(self.label_name)
        df = df.dropna(subset=cleaning_columns)
        if "NIVEAU DE SECURITE" in df.columns:
            df = df.drop(columns=["NIVEAU DE SECURITE"])
        return df

    def get_month(self, df: pd.Series) -> pd.Series:
        return df.apply(lambda x: x.month)

    def get_week(self, df: pd.Series) -> pd.Series:
        return df.apply(lambda x: x.week)

    def get_start_hour(self, df):
        return df.apply(lambda x: x[:-2])

    def __get_dict_of_average_plane_by_day(self, df: pd.DataFrame, airport_type: str) -> Dict:
        min_date = df["DATE"].min()
        max_date = df["DATE"].max()
        number_of_days = (max_date - min_date).days + 1
        return (
            df[[airport_type, "IDENTIFIANT", "DATE"]]
            .groupby([airport_type, "DATE"])
            .count()
            .reset_index()[[airport_type, "IDENTIFIANT"]]
            .groupby([airport_type])
            .sum()
            .apply(lambda x: x / number_of_days)["IDENTIFIANT"]
            .to_dict()
        )

    def get_average_plane_take_off_or_landing_by_day(
        self, df: pd.DataFrame, airport_type
    ) -> pd.DataFrame:
        average_nb_plane_by_day = self.__get_dict_of_average_plane_by_day(
            df, airport_type
        )
        return df[airport_type].apply(lambda x: average_nb_plane_by_day[x])

    def split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = df[self.training_columns]
        y = df[self.label_name]
        return X, y

    def fit_transform_dummify_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_dummify is not None:
            return self.label_encoder.fit_transform(X)
        return X

    def transform_dummify_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_dummify is not None:
            return self.label_encoder.transform(X)
        return X

    def fit(self, dataframe: pd.DataFrame):
        df = dataframe.copy()

        df = self.cleaning(df)

        df.loc[:, "DATE"] = pd.to_datetime(df["DATE"])
        df.loc[:, "MOIS"] = self.get_month(df["DATE"])
        df.loc[:, "SEMAINE"] = self.get_week(df["DATE"])

        X, y = self.split_X_y(df)

        X = self.fit_transform_dummify_columns(X)

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "DATE"] = pd.to_datetime(df["DATE"])
        df.loc[:, "MOIS"] = self.get_month(df["DATE"])
        df.loc[:, "SEMAINE"] = self.get_week(df["DATE"])

        df = df[self.training_columns]
        X = self.transform_dummify_columns(df)
        return X

    def save_feature_engineering(self, path=None):
        """
        Save to file in the current working directory
        """
        if path is None:
            path = "../../data/output/feature_engineering.pkl"
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_feature_engineering(self, path=None):
        """
        Load file in an instance
        """
        if path is None:
            path = "../../data/output/feature_engineering.pkl"
        with open(path, "rb") as file:
            pickle_fe = pickle.load(file)

        return pickle_fe
