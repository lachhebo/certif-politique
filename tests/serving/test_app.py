from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from pandas._testing import assert_frame_equal

from certifia.feature_engineering import FeatureEngineering
from certifia.serving.app import load_csv, prediction
from certifia.training import Training

TESTED_MODULE = 'certifia.serving.app'
FEATURE_ENGINEERING_MODULE = "certifia.feature_engineering"
TRAINING_MODULE = "certifia.training"


class TestDataAccess(TestCase):
    @patch('pandas.read_csv')
    def test_load_csv(self, mock_read_csv):
        # Given
        file = 'vol_test.csv'
        # When
        load_csv(file=file)
        # Then
        self.assertEqual(mock_read_csv.call_count, 1)

    def test_prediction_return_df_with_only_identifiant_when_retard_arrivee_is_none(self):
        # Given
        df = pd.DataFrame(columns=['IDENTIFIANT'])
        form = {'retard_arrivee': None}
        # When
        result = prediction(df, form)
        # Then
        assert_frame_equal(result, df)

    @patch(f'{FEATURE_ENGINEERING_MODULE}.FeatureEngineering.load_feature_engineering')
    @patch(f'{FEATURE_ENGINEERING_MODULE}.FeatureEngineering.transform')
    @patch(f'{TRAINING_MODULE}.Training.load_model')
    @patch(f'{TRAINING_MODULE}.Training.predict')
    def test_prediction_return_df_with_prediction_when_retard_arrivee_is_on(
            self,
            mock_predict,
            mock_load_model,
            mock_transform_feat_engineering,
            mock_load_feature_engineering
    ):
        # Given
        df = pd.DataFrame(columns=['IDENTIFIANT'])
        form = {'retard_arrivee': 'on'}
        expected_df = pd.DataFrame(columns=['IDENTIFIANT', "PREDICTION RETARD A L'ARRIVEE"])
        mock_load_feature_engineering.return_value = FeatureEngineering()
        mock_transform_feat_engineering.return_value = pd.DataFrame(columns=['IDENTIFIANT', 'DATE'])
        mock_load_model.return_value = Training()
        mock_predict.return_value = pd.Series(name='PREDICTION', dtype=object)

        # When
        result = prediction(df, form)

        # Then
        assert_frame_equal(result, expected_df)
        self.assertEqual(mock_load_feature_engineering.call_count, 1)
        self.assertEqual(mock_transform_feat_engineering.call_count, 1)
        self.assertEqual(mock_load_model.call_count, 1)
        self.assertEqual(mock_predict.call_count, 1)
