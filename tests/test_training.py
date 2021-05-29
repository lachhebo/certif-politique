from unittest.mock import patch
from unittest import TestCase

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal
from sklearn.ensemble import RandomForestRegressor

from certifia.training import Training

TESTED_MODULE = 'certifia.training'
LOGGER_MODULE = 'certifia.utils.logger'


class TestTraining(TestCase):
    def test_instantiate_training_should_return_the_encoder(self):
        # given
        # when
        result = Training()

        # then
        self.assertIsInstance(result.rf_regressor, RandomForestRegressor)

    @patch('sklearn.ensemble.RandomForestRegressor.fit')
    def test_fit_should_train_algorithm(self, mock_rf_fit):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        y = pd.Series(name="RETARD A L'ARRIVEE", dtype=int)

        # when
        result = Training().fit(X, y)

        # then
        self.assertEqual(mock_rf_fit.call_count, 1)

    @patch('sklearn.ensemble.RandomForestRegressor.predict')
    def test_predict_should_return_prediction(self, mock_rf_predict):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        mock_rf_predict.return_value = [0]
        expected = [0]

        # when
        result = Training().predict(X)

        # then
        self.assertEqual(mock_rf_predict.call_count, 1)
        self.assertEqual(result, expected)

    @patch('logging.Logger.info')
    @patch(f'{TESTED_MODULE}.Training.predict')
    def test_score_should_logged_metrics(self, mock_training_pred, mock_logger):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        y = pd.Series(name="RETARD A L'ARRIVEE", data=[0, 0])
        mock_training_pred.return_value = [0, 0]

        # when
        Training().score(X, y)

        # then
        self.assertEqual(mock_logger.call_count, 4)

        # TODO add test for saving the model
