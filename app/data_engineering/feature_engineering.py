import pickle

import pandas as pd

from app import ROOT_PATH
from app.data_engineering.utils import get_month, get_week, get_hour
from app.ml.multi_column_label_encode import MultiColumnLabelEncoder


class FeatureEngineering:
    def __init__(self, training_columns=None, columns_to_dummify=None):
        self.training_columns = training_columns
        self.columns_to_dummify = columns_to_dummify
        self.label_encoder = MultiColumnLabelEncoder(
            encoded_columns=self.columns_to_dummify
        )
        self.average_nb_plane_by_day = {}

    def __get_dict_of_average_plane_by_day(self, df, airport_type: str):
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

    def get_average_plane_take_off_or_landing_by_day(self, df, airport_type):
        self.average_nb_plane_by_day[
            airport_type
        ] = self.__get_dict_of_average_plane_by_day(df, airport_type)
        return df[airport_type].apply(
            lambda x: self.average_nb_plane_by_day[airport_type][x]
        )

    def apply_average_plane_take_off_or_landing_by_day(self, df, airport_type):
        return df[airport_type].apply(
            lambda x: self.average_nb_plane_by_day[airport_type][x]
            if x in self.average_nb_plane_by_day[airport_type]
            else 0
        )

    def keep_training_columns(self, X):
        if self.training_columns is not None:
            return X[self.training_columns]
        return X

    def fit(self, dataframe: pd.DataFrame):
        X = dataframe.copy()

        X.loc[:, "DATE"] = pd.to_datetime(X["DATE"])
        X.loc[:, "MOIS"] = get_month(X["DATE"])
        X.loc[:, "SEMAINE"] = get_week(X["DATE"])
        X.loc[:, "HEURE DEPART PROGRAMME"] = get_hour(X["DEPART PROGRAMME"])
        X.loc[:, "HEURE ARRIVEE PROGRAMMEE"] = get_hour(X["ARRIVEE PROGRAMMEE"])

        X.loc[
            :, "NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR"
        ] = self.get_average_plane_take_off_or_landing_by_day(X, "AEROPORT DEPART")
        X.loc[
            :, "NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR"
        ] = self.get_average_plane_take_off_or_landing_by_day(X, "AEROPORT ARRIVEE")

        X = self.label_encoder.fit_transform(X)

        X = self.keep_training_columns(X)

        return X

    def transform(self, dataframe: pd.DataFrame):
        X = dataframe.copy()

        X.loc[:, "DATE"] = pd.to_datetime(X["DATE"])
        X.loc[:, "MOIS"] = get_month(X["DATE"])
        X.loc[:, "SEMAINE"] = get_week(X["DATE"])
        X.loc[:, "HEURE DEPART PROGRAMME"] = get_hour(X["DEPART PROGRAMME"])
        X.loc[:, "HEURE ARRIVEE PROGRAMMEE"] = get_hour(X["ARRIVEE PROGRAMMEE"])

        X.loc[
            :, "NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR"
        ] = self.apply_average_plane_take_off_or_landing_by_day(X, "AEROPORT DEPART")
        X.loc[
            :, "NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR"
        ] = self.apply_average_plane_take_off_or_landing_by_day(X, "AEROPORT ARRIVEE")
        X = self.label_encoder.transform(X)

        X = self.keep_training_columns(X)

        return X

    def save_feature_engineering(self, path=None):
        """
        Save to file in the current working directory
        """
        if path is None:
            feature_eng_path = ROOT_PATH / "data" / "output" / "feature_engineering.pkl"
            path = feature_eng_path.resolve()
        with open(path, "wb") as file:
            pickle.dump(self, file)
