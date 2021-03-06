from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from app.data_engineering.data_cleaning import DataCleaning
from app.data_engineering.feature_engineering import FeatureEngineering
from app.ml.model import Model
from app.ml.utils import load_csv, prediction

TESTED_MODULE = 'app.ml.model'
FEATURE_ENGINEERING_MODULE = "app.data_engineering.feature_engineering"
TRAINING_MODULE = "app.ml.model"


class TestMLUtils(TestCase):
    @patch('pandas.read_csv')
    def test_load_csv(self, mock_read_csv):
        # Given
        file = 'vol_test.csv'
        # When
        load_csv(file=file)
        # Then
        assert mock_read_csv.call_count == 1

    def test_prediction_return_df_with_only_identifiant_when_retard_arrivee_is_none(self):
        # Given
        df = pd.DataFrame(columns=['IDENTIFIANT'])
        form = {'retard_arrivee': None}
        # When
        result = prediction(df, form)
        # Then
        assert_frame_equal(result, df)

    @patch('app.data_engineering.data_cleaning.DataCleaning.transform')
    @patch('app.data_engineering.data_cleaning.DataCleaning.load_cleaner')
    @patch('app.ml.utils.load_feature_engineering')
    @patch(f'{FEATURE_ENGINEERING_MODULE}.FeatureEngineering.transform')
    @patch(f'{TRAINING_MODULE}.Model.load_model')
    @patch(f'{TRAINING_MODULE}.Model.predict')
    def test_prediction_return_df_with_prediction_when_retard_arrivee_is_on(
            self,
            mock_predict,
            mock_load_model,
            mock_transform_feat_engineering,
            mock_load_feature_engineering,
            mock_load_cleaner,
            mock_transform,
    ):
        # Given
        df = pd.DataFrame(columns=['IDENTIFIANT'])
        form = {'retard_arrivee': 'on'}
        expected_df = pd.DataFrame(columns=['IDENTIFIANT', "PREDICTION RETARD A L'ARRIVEE"])
        mock_load_cleaner.return_value = DataCleaning(['IDENTIFIANT'], "RETARD A L'ARRIVEE")
        mock_load_feature_engineering.return_value = FeatureEngineering()
        mock_transform_feat_engineering.return_value = pd.DataFrame(columns=['IDENTIFIANT', 'DATE'])
        mock_load_model.return_value = Model()
        mock_predict.return_value = pd.Series(name='PREDICTION', dtype=object)

        # When
        result = prediction(df, form)

        # Then
        assert_frame_equal(result, expected_df)
        assert mock_load_feature_engineering.call_count == 1
        assert mock_transform_feat_engineering.call_count == 1
        assert mock_load_model.call_count == 1
        assert mock_predict.call_count == 1
        assert mock_load_cleaner.call_count == 1
        assert mock_transform.call_count == 1
