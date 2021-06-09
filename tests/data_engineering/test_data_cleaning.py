from unittest import TestCase
from unittest.mock import patch, mock_open

import pandas as pd
from pandas.testing import assert_frame_equal

from app.data_engineering.data_cleaning import DataCleaning


class TestDataCleaning(TestCase):
    def test_cleaning_remove_unused_columns(self):
        # given
        features_columns = ['feat1', 'feat2']
        label = 'target'

        data_cleaning = DataCleaning(features_columns, label)
        df = pd.DataFrame({
            'feat1': [0, 1, 2],
            'feat2': [0, pd.NaT, 2],
            'feat3': [0, pd.NaT, 2],
            'target': [0, 1, 2]
        })

        expected_df = pd.DataFrame({
            'feat1': [0, 2],
            'feat2': [0, 2],
            'feat3': [0, 2],
            'target': [0, 2]
        })

        # when
        output_df = data_cleaning.drop_na(df)

        # then
        assert_frame_equal(expected_df.reset_index(drop=True), output_df.reset_index(drop=True), check_dtype=False)

    @patch('pickle.dump')
    def test_save_data_cleaning_should_call_open_methof_and_save_cleaner(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            DataCleaning([], '').save_cleaner()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)

    @patch('pickle.load')
    def test_load_data_cleaning_should_call_open_method_and_return_a_data_cleaning_instance(self, mock_pickle):
        # Given
        with patch('builtins.open', mock_open()) as mock_open_method:
            # When
            DataCleaning([], '').load_cleaner()

        # Then
        self.assertEqual(mock_open_method.call_count, 1)
        self.assertEqual(mock_pickle.call_count, 1)
