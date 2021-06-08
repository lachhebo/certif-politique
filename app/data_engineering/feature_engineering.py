import pickle

import pandas as pd

from app import ROOT_PATH
from app.data_engineering.utils import get_dict_of_average_plane_by_day, parse_date, apply_sqrt, get_airport_dict
from app.ml.multi_column_label_encode import MultiColumnLabelEncoder


class FeatureEngineering:
    def __init__(self, training_columns=None, columns_to_dummify=None, df_airport=None):
        self.training_columns = training_columns
        self.columns_to_dummify = columns_to_dummify
        self.label_encoder = MultiColumnLabelEncoder(
            encoded_columns=self.columns_to_dummify
        )
        self.average_nb_plane_by_day = {}
        self.airport = get_airport_dict(df_airport)

    def get_average_plane_take_off_or_landing_by_day(self, df, airport_type):
        self.average_nb_plane_by_day[
            airport_type
        ] = get_dict_of_average_plane_by_day(df, airport_type)
        return df[airport_type].apply(
            lambda x: self.average_nb_plane_by_day[airport_type][x]
        )

    def apply_average_plane_take_off_or_landing_by_day(self, df, airport_type):
        return df[airport_type].apply(
            lambda x: self.average_nb_plane_by_day[airport_type][x]
            if x in self.average_nb_plane_by_day[airport_type]
            else 0
        )

    def add_data_from_airport(self, X):
        if not self.airport:
            return X
        X.loc[:, 'PAYS DEPART'] = X['AEROPORT DEPART'].apply(lambda x: self.airport.get(x)['PAYS'])
        X.loc[:, 'PAYS ARRIVEE'] = X['AEROPORT ARRIVEE'].apply(lambda x: self.airport.get(x)['PAYS'])

        X.loc[:, 'HAUTEUR DEPART'] = X['AEROPORT DEPART'].apply(lambda x: self.airport.get(x)['HAUTEUR'])
        X.loc[:, 'HAUTEUR ARRIVEE'] = X['AEROPORT ARRIVEE'].apply(lambda x: self.airport.get(x)['HAUTEUR'])
        X.loc[:, 'LONGITUDE ARRIVEE'] = X['AEROPORT ARRIVEE'].apply(lambda x: self.airport.get(x)['LONGITUDE TRONQUEE'])
        X.loc[:, 'LATITUDE ARRIVEE'] = X['AEROPORT ARRIVEE'].apply(lambda x: self.airport.get(x)['LATITUDE TRONQUEE'])

        X.loc[:, 'PRIX RETARD PREMIERE 20 MINUTES'] = X['AEROPORT ARRIVEE'].apply(
            lambda x: self.airport.get(x)['PRIX RETARD PREMIERE 20 MINUTES'])
        X.loc[:, 'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'] = X['AEROPORT ARRIVEE'].apply(
            lambda x: self.airport.get(x)['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'])
        return X

    def keep_training_columns(self, X):
        if self.training_columns is not None:
            return X[self.training_columns]
        return X

    def fit(self, dataframe: pd.DataFrame):
        X = parse_date(dataframe)

        X.loc[
            :, "NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR"
        ] = self.get_average_plane_take_off_or_landing_by_day(X, "AEROPORT DEPART")
        X.loc[
            :, "NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR"
        ] = self.get_average_plane_take_off_or_landing_by_day(X, "AEROPORT ARRIVEE")

        X['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE'] = apply_sqrt(
            X['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE'])
        X["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"] = apply_sqrt(
            X["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"])

        X = self.add_data_from_airport(X)

        X = self.label_encoder.fit_transform(X)

        X = self.keep_training_columns(X)

        return X

    def transform(self, dataframe: pd.DataFrame):
        X = parse_date(dataframe)

        X.loc[
            :, "NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR"
        ] = self.apply_average_plane_take_off_or_landing_by_day(X, "AEROPORT DEPART")
        X.loc[
            :, "NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR"
        ] = self.apply_average_plane_take_off_or_landing_by_day(X, "AEROPORT ARRIVEE")

        X['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE'] = apply_sqrt(
            X['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE'])
        X["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"] = apply_sqrt(
            X["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"])

        X = self.add_data_from_airport(X)

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
