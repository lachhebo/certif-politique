from datetime import date
from unittest import TestCase
from unittest.mock import patch, mock_open, Mock

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from app.data_engineering.feature_engineering import FeatureEngineering
from app.ml.multi_column_label_encode import MultiColumnLabelEncoder

TESTED_MODULE = 'app.data_engineering.feature_engineering'
UTILS_MODULE = 'app.data_engineering.utils'


class TestFeatureEngineering(TestCase):
    def test_instantiate_feature_engineering_should_return_itself(self):
        # given
        # when
        result = FeatureEngineering()

        # then
        self.assertEqual(result.training_columns, None)
        self.assertEqual(result.columns_to_dummify, None)
        self.assertEqual(result.average_nb_plane_by_day, {})
        self.assertIsInstance(result.label_encoder, MultiColumnLabelEncoder)

    @patch(f'{TESTED_MODULE}.MultiColumnLabelEncoder.fit_transform')
    @patch('pandas.to_datetime')
    @patch(f'{UTILS_MODULE}.get_month')
    @patch(f'{UTILS_MODULE}.get_week')
    @patch(f'{UTILS_MODULE}.get_hour')
    @patch(f'{TESTED_MODULE}.apply_sqrt')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.get_average_plane_take_off_or_landing_by_day')
    def test_fit_call_multiple_feature_engineering_method(
            self,
            mock_get_average_plane_take_off_or_landing_by_day,
            mock_apply_sqrt,
            mock_get_hour,
            mock_get_week,
            mock_get_month,
            mock_to_datetime,
            mock_fit_transform
    ):
        # given
        df = pd.DataFrame(columns=[
            "DATE",
            "DEPART PROGRAMME",
            "ARRIVEE PROGRAMMEE",
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"
        ])
        mock_fit_transform.return_value = pd.DataFrame, pd.DataFrame

        # when
        FeatureEngineering().fit(df)

        # then
        self.assertEqual(mock_to_datetime.call_count, 1)
        self.assertEqual(mock_get_month.call_count, 1)
        self.assertEqual(mock_get_week.call_count, 1)
        self.assertEqual(mock_get_hour.call_count, 2)
        self.assertEqual(mock_apply_sqrt.call_count, 2)
        self.assertEqual(mock_get_average_plane_take_off_or_landing_by_day.call_count, 2)

    @patch('pandas.to_datetime')
    @patch(f'{UTILS_MODULE}.get_month')
    @patch(f'{UTILS_MODULE}.get_week')
    @patch(f'{UTILS_MODULE}.get_hour')
    @patch(f'{TESTED_MODULE}.apply_sqrt')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.apply_average_plane_take_off_or_landing_by_day')
    @patch(f'{TESTED_MODULE}.MultiColumnLabelEncoder')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.keep_training_columns')
    def test_transform_should_return_dataframe_with_a_call_of_self_transform_dummify(
            self,
            mock_keep_training_columns,
            mock_transform_dummify_columns,
            mock_apply_average_plane_take_off_or_landing_by_day,
            mock_apply_sqrt,
            mock_get_hour,
            mock_get_week,
            mock_get_month,
            mock_to_datetime,
    ):
        # given
        training_columns = [
            "DATE",
            "DEPART PROGRAMME",
            "ARRIVEE PROGRAMMEE",
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"
        ]
        df = pd.DataFrame(columns=training_columns)
        mymock = Mock()
        mock_transform_dummify_columns.fit_transform = mymock
        mymock.return_value = pd.DataFrame, pd.DataFrame

        # when
        FeatureEngineering(training_columns).transform(df)

        # then
        self.assertEqual(mock_to_datetime.call_count, 1)
        self.assertEqual(mock_get_month.call_count, 1)
        self.assertEqual(mock_get_week.call_count, 1)
        self.assertEqual(mock_get_hour.call_count, 2)
        self.assertEqual(mock_apply_sqrt.call_count, 2)
        self.assertEqual(mock_apply_average_plane_take_off_or_landing_by_day.call_count, 2)
        self.assertEqual(mock_transform_dummify_columns.call_count, 1)
        self.assertEqual(mock_keep_training_columns.call_count, 1)

    def test_average_plane_by_return_series_of_number_of_takeoff(self):
        training_columns = ['AEROPORT', "IDENTIFIANT", "DATE"]
        df = pd.DataFrame(
            columns=training_columns,
            data=[['A', 0, date(2020, 1, 1)],
                  ['A', 1, date(2020, 1, 1)],
                  ['A', 2, date(2020, 1, 2)],
                  ['B', 3, date(2020, 1, 2)],
                  ['B', 4, date(2020, 1, 3)],
                  ['C', 5, date(2020, 1, 4)]])
        expected_df = pd.Series(
            name="AEROPORT",
            data=[0.75, 0.75, 0.75, 0.5, 0.5, 0.25])

        # when
        result = FeatureEngineering(training_columns).get_average_plane_take_off_or_landing_by_day(df, 'AEROPORT')

        # then
        assert_series_equal(result, expected_df)

    @patch('pickle.dump')
    def test_save_feature_engineering(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            FeatureEngineering().save_feature_engineering()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)

    def test_add_data_from_airport(self):
        # Given
        df_vols = pd.DataFrame(
            columns=['AEROPORT DEPART', 'AEROPORT ARRIVEE'],
            data=[['AIR', 'AIR'],
                  ['ISM', 'RAI'],
                  ['AIR', 'ISM']]
        )
        df_airport = pd.DataFrame(
            columns=[
                'CODE IATA',
                'LONGITUDE',
                'LATITUDE',
                'HAUTEUR',
                'PRIX RETARD PREMIERE 20 MINUTES',
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES',
                'PAYS',
            ],
            data=[
                ['AIR', '45.1', '45.1', 10, 1,  2, 'FR'],
                ['AIR', '45.1', '45.1', 10, 1,  2, 'FR'],
                ['RAI',   '12', '16.6', 20, 1,  2, 'EN'],
                ['RAI',   '12', '16.6', 20, 3, 10, 'EN'],
                ['ISM',   '-1',   '10', 5, 1,  2, 'ME'],
            ]
        )
        expected = pd.DataFrame(
            columns=[
                'AEROPORT DEPART',
                'AEROPORT ARRIVEE',
                'PAYS DEPART',
                'PAYS ARRIVEE',
                'HAUTEUR DEPART',
                'HAUTEUR ARRIVEE',
                'LONGITUDE ARRIVEE',
                'LATITUDE ARRIVEE',
                'PRIX RETARD PREMIERE 20 MINUTES',
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'
            ],
            data=[['AIR', 'AIR', 'FR', 'FR', 10, 10, 45, 45, 1, 2],
                  ['ISM', 'RAI', 'ME', 'EN',  5, 20, 12, 17, 2, 6],
                  ['AIR', 'ISM', 'FR', 'ME', 10,  5, -1, 10, 1, 2]]
        )

        # When
        result = FeatureEngineering(df_airport=df_airport).add_data_from_airport(df_vols)
        # Then
        assert_frame_equal(result, expected)

    def test_fit_return_features_and_labels_dataframe(self):
        # given
        df_columns = [
            'IDENTIFIANT',
            'DEPART PROGRAMME',
            'ARRIVEE PROGRAMMEE',
            'AEROPORT DEPART',
            'AEROPORT ARRIVEE',
            'DATE',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"
        ]
        training_columns = [
            'AEROPORT DEPART',
            'MOIS',
            'SEMAINE',
            'HEURE DEPART PROGRAMME',
            'HEURE ARRIVEE PROGRAMMEE',
            'NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR',
            'NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR',
        ]
        dummi_columns = ['AEROPORT DEPART', 'AEROPORT ARRIVEE']
        df = pd.DataFrame(
            columns=df_columns,
            data=[
                [1, 1100, 1200, 'C', 'A', '12-29-2019', 1, 1],
                [2, 1130, 1330, 'B', 'A', '12-30-2019', 1, 1],
                [3, 2059, 2100, 'B', 'C', '12-30-2019', 1, 1]]
        )
        expected_X = pd.DataFrame(columns=training_columns, data=[
            [1, 12, 52, 11, 12, 0.5, 1],
            [0, 12, 1, 11, 13, 1, 1],
            [0, 12, 1, 20, 21, 1, 0.5]])

        # when
        result_X = FeatureEngineering(training_columns, dummi_columns).fit(df)

        # then
        assert_frame_equal(result_X, expected_X, check_dtype=False)

    def test_transform_return_features_dataframe(self):
        # given
        df_columns = [
            'IDENTIFIANT',
            'DEPART PROGRAMME',
            'ARRIVEE PROGRAMMEE',
            'AEROPORT DEPART',
            'AEROPORT ARRIVEE',
            'DATE',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
        ]
        training_columns = [
            'MOIS',
            'SEMAINE',
            'HEURE DEPART PROGRAMME',
            'HEURE ARRIVEE PROGRAMMEE',
            'NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR',
            'NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
        ]
        df = pd.DataFrame(
            columns=df_columns,
            data=[
                [1, 1100, 1200, 'C', 'A', '12-29-2019', 1, 4],
                [2, 1130, 1330, 'B', 'A', '12-30-2019', 4, 9],
                [3, 2059, 2100, 'B', 'C', '12-30-2019', 9, 16]]
        )
        expected_df = pd.DataFrame(columns=training_columns, data=[
            [12, 52, 11, 12, 0, 0, 1, 2],
            [12,  1, 11, 13, 0, 0, 2, 3],
            [12,  1, 20, 21, 0, 0, 3, 4]])

        feature_engineering = FeatureEngineering(training_columns)
        feature_engineering.average_nb_plane_by_day = {'AEROPORT DEPART': {}, 'AEROPORT ARRIVEE': {}}

        # when
        result = feature_engineering.transform(df)

        # then
        assert_frame_equal(result, expected_df, check_dtype=False)

    def test_fit_transform_return_features_dataframe(self):
        # given
        columns = [
            'IDENTIFIANT',
            'DEPART PROGRAMME',
            'ARRIVEE PROGRAMMEE',
            'AEROPORT DEPART',
            'AEROPORT ARRIVEE',
            'DATE',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
        ]
        training_columns = [
            'MOIS',
            'SEMAINE',
            'HEURE DEPART PROGRAMME',
            'HEURE ARRIVEE PROGRAMMEE',
            'NOMBRE DECOLLAGE PAR AEROPORT PAR JOUR',
            'NOMBRE ATTERRISSAGE PAR AEROPORT PAR JOUR',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
        ]
        train_df = pd.DataFrame(
            columns=columns,
            data=[
                [1, 1100, 1200, 'C', 'A', '12-29-2019',  1,  4],
                [2, 1130, 1330, 'B', 'A', '12-30-2019',  9, 16],
                [3, 2059, 2100, 'B', 'C', '12-30-2019', 25, 36]]
        )
        test_df = pd.DataFrame(
            columns=columns,
            data=[
                [1,  35,  130, 'A', 'C', '01-01-2020', 1, 4],
                [2, 800, 2359, 'C', 'A', '01-01-2020', 9, 16]]
        )
        expected_df = pd.DataFrame(columns=training_columns, data=[
            [1, 1, 0,  1, 0, 0.5, 1, 2],
            [1, 1, 8, 23, 0.5, 1, 3, 4]])

        feature_engineering = FeatureEngineering(training_columns)

        # when
        feature_engineering.fit(train_df)
        result = feature_engineering.transform(test_df)

        # then
        assert_frame_equal(result, expected_df, check_dtype=False)
