from datetime import datetime
from unittest import TestCase
from unittest.mock import patch, mock_open

import pandas as pd
import numpy as np
from pandas._testing import assert_series_equal

from app.data_engineering.utils import load_feature_engineering, get_airport_dict, remove_unused_columns, get_hour, \
    apply_sqrt, get_week, get_month


class TestDataEngineeringUtils(TestCase):
    @patch('pickle.load')
    def test_load_feature_engineering(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            load_feature_engineering()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)

    def test_get_airport_dict_return_empty_dict_when_pass_empty_df(self):
        # Given
        df_airport = None
        # When
        result = get_airport_dict(df_airport)
        # Then
        self.assertDictEqual(result, {})

    def test_get_airport_dict_return_dict_with_airport_when_given_airport_df(self):
        # Given
        df_airport = pd.DataFrame(
            columns=[
                'CODE IATA',
                'LONGITUDE',
                'LATITUDE',
                'HAUTEUR',
                'PRIX RETARD PREMIERE 20 MINUTES',
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES',
            ],
            data=[
                ['AIR', '45.1', '45.1', 10, 1, 2],
                ['AIR', '45.1', '45.1', 10, 1, 2],
                ['RAI', '12', '16.6', 10, 1, 2],
                ['RAI', '12', '16.6', 10, 3, 10],
                ['ISM', '-1', '10', 10, 1, 2],
            ]
        )
        expected = {
            'AIR': {
                'LONGITUDE': 45.1,
                'LATITUDE': 45.1,
                'HAUTEUR': 10,
                'PRIX RETARD PREMIERE 20 MINUTES': 1,
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 2,
                'LONGITUDE TRONQUEE': 45,
                'LATITUDE TRONQUEE': 45
            },
            'RAI': {
                'LONGITUDE': 12.0,
                'LATITUDE': 16.6,
                'HAUTEUR': 10,
                'PRIX RETARD PREMIERE 20 MINUTES': 2,
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 6,
                'LONGITUDE TRONQUEE': 12,
                'LATITUDE TRONQUEE': 17
            },
            'ISM': {
                'LONGITUDE': -1.0,
                'LATITUDE': 10.0,
                'HAUTEUR': 10,
                'PRIX RETARD PREMIERE 20 MINUTES': 1,
                'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 2,
                'LONGITUDE TRONQUEE': -1,
                'LATITUDE TRONQUEE': 10
            }
        }

        # When
        result = get_airport_dict(df_airport)
        # Then
        self.assertDictEqual(result, expected)

    def test_remove_unused_columns(self):
        # Given
        df = pd.DataFrame(columns=['NIVEAU DE SECURITE'], data=[[10], [10]])
        # When
        result = remove_unused_columns(df)
        # then
        np.array_equal(result.values, pd.DataFrame().values)

    def test_get_week(self):
        # Given
        df = pd.Series(data=[datetime(2020, 1, 1), datetime(2020, 2, 1)])
        expected = pd.Series(data=[1, 5])
        # When
        result = get_week(df)
        # then
        assert_series_equal(result, expected, check_names=False)

    def test_get_month(self):
        # Given
        df = pd.Series(data=[datetime(2020, 1, 1), datetime(2020, 2, 1)])
        expected = pd.Series(data=[1, 2])
        # When
        result = get_month(df)
        # then
        assert_series_equal(result, expected, check_names=False)

    def test_get_hour(self):
        # Given
        df = pd.Series(data=[555, 2201])
        expected = pd.Series(data=[5, 22])
        # When
        result = get_hour(df)
        # then
        assert_series_equal(result, expected, check_names=False)

    def test_apply_sqrt(self):
        # Given
        df = pd.Series(data=[4, 16])
        expected = pd.Series(data=[2., 4.])
        # When
        result = apply_sqrt(df)
        # then
        assert_series_equal(result, expected, check_names=False)
