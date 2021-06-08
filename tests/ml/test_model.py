from unittest import TestCase
from unittest.mock import patch, mock_open

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.ml.model import Model

TESTED_MODULE = 'app.ml.model'


class TestModel(TestCase):
    def test_instantiate_training_should_return_the_encoder(self):
        # given
        # when
        result = Model()

        # then
        self.assertIsInstance(result.model, RandomForestRegressor)

    @patch('sklearn.ensemble.RandomForestRegressor.fit')
    def test_fit_should_train_algorithm(self, mock_rf_fit):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        y = pd.Series(name="RETARD A L'ARRIVEE", dtype=int)

        # when
        Model().fit(X, y)

        # then
        self.assertEqual(mock_rf_fit.call_count, 1)

    @patch('sklearn.ensemble.RandomForestRegressor.predict')
    def test_predict_should_return_prediction(self, mock_rf_predict):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        mock_rf_predict.return_value = [0]
        expected = [0]

        # when
        result = Model().predict(X)

        # then
        self.assertEqual(mock_rf_predict.call_count, 1)
        self.assertEqual(result, expected)

    @patch('app.utils.logger.Logger.info')
    @patch(f'{TESTED_MODULE}.Model.predict')
    def test_score_should_logged_metrics(self, mock_training_pred, mock_logger):
        # given
        X = pd.DataFrame(columns=["Col_Train_1", "Col_Train_2"], dtype=int)
        y = pd.Series(name="RETARD A L'ARRIVEE", data=[0, 0])
        mock_training_pred.return_value = [0, 0]

        # when
        Model().compute_metrics(X, y)

        # then
        self.assertEqual(mock_logger.call_count, 4)

    @patch('pickle.dump')
    def test_save_model(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            Model().save_model()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)

    @patch('pickle.load')
    def test_load_model(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            Model().load_model()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)
