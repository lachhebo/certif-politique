from unittest.mock import patch
from unittest import TestCase
import pandas as pd
from pandas._testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder

from certifia.utils.multi_column_label_encode import MultiColumnLabelEncoder

TESTED_MODULE = 'certifia.utils.multi_column_label_encode'


class TestMultiColumnLabelEncoder(TestCase):
    def test_instantiate_encoder_should_return_the_encoder(self):
        # given
        # when
        result = MultiColumnLabelEncoder()
        # then
        self.assertEqual(result.columns, None)
        self.assertEqual(result.label_encoder, {})

    def test_instantiate_encoder_with_list_of_columns_should_return_the_encoder(self):
        # given
        columns = ["Col1", "Col2"]
        # when
        result = MultiColumnLabelEncoder(columns=columns)
        # then
        self.assertEqual(result.columns, columns)

    def test_encoder_fit_should_return_the_encoder_with_label_encoder_filled(self):
        # given
        columns = ["Col1", "Col2"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A'], ['B', 'B'], ['A', 'C']])
        # when
        result = MultiColumnLabelEncoder(columns).fit(df)
        # then
        self.assertEqual(len(result.label_encoder.keys()), 2)
        self.assertIsInstance(result.label_encoder['Col1'], LabelEncoder)
        self.assertIsInstance(result.label_encoder['Col2'], LabelEncoder)

    def test_encoder_fit_should_return_the_encoder_with_label_encoder_filled_for_given_column_list(self):
        # given
        columns = ["Col1", "Col2", "Col3"]
        dummy_columns = ["Col1", "Col3"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A', 'Alpha'], ['B', 'B', 'Beta'], ['A', 'C', 'Gamma']])
        # when
        result = MultiColumnLabelEncoder(dummy_columns).fit(df)
        # then
        self.assertEqual(len(result.label_encoder.keys()), 2)
        self.assertIsInstance(result.label_encoder['Col1'], LabelEncoder)
        self.assertIsInstance(result.label_encoder['Col3'], LabelEncoder)

    @patch('sklearn.preprocessing.LabelEncoder.transform')
    def test_encoder_transform_should_return_the_dataframe_with_all_columns_encoded(self, mock_label_encod):
        # given
        mock_label_encod.return_value = [0, 1, 2]
        columns = ["Col1", "Col2"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A'], ['B', 'B'], ['A', 'C']])
        expected_df = pd.DataFrame(columns=columns, data=[[0, 0], [1, 1], [2, 2]])
        # when
        result = MultiColumnLabelEncoder(columns).fit(df).transform(df)
        # then
        assert_frame_equal(result, expected_df)

    @patch('sklearn.preprocessing.LabelEncoder.transform')
    def test_encoder_transform_should_return_the_dataframe_with_encoded_columns_for_column_list(self, mock_label_encod):
        # given
        mock_label_encod.return_value = [1, 0, 1]
        columns = ["Col1", "Col2", "Col3"]
        dummy_columns = ["Col1", "Col2"]
        df = pd.DataFrame(columns=columns, data=[['B', 'B', 'Alpha'], ['A', 'A', 'Beta'], ['B', 'B', 'Gamma']])
        expected_df = pd.DataFrame(columns=columns, data=[[1, 1, 'Alpha'], [0, 0, 'Beta'], [1, 1, 'Gamma']])
        # when
        result = MultiColumnLabelEncoder(dummy_columns).fit(df).transform(df)
        # then
        assert_frame_equal(result, expected_df)

    # Integration tests
    def test_encoder_fittransform_should_return_the_dataframe_with_all_columns_encoded(self):
        # given
        columns = ["Col1", "Col2"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A'], ['B', 'B'], ['A', 'C']])
        expected_df = pd.DataFrame(columns=columns, data=[[0, 0], [1, 1], [0, 2]])
        # when
        result = MultiColumnLabelEncoder('all').fit_transform(df)
        # then
        assert_frame_equal(result, expected_df)

    def test_encoder_fittransform_should_return_the_dataframe_encoded_for_column_list(self):
        # given
        columns = ["Col1", "Col2", "Col3"]
        dummy_columns = ["Col1", "Col2"]
        df = pd.DataFrame(columns=columns, data=[['A', 'B', 'Alpha'], ['Z', 'A', 'Beta'], ['C', 'D', 'Gamma']])
        expected_df = pd.DataFrame(columns=columns, data=[[0, 1, 'Alpha'], [2, 0, 'Beta'], [1, 2, 'Gamma']])
        # when
        result = MultiColumnLabelEncoder(dummy_columns).fit_transform(df)
        # then
        assert_frame_equal(result, expected_df)
