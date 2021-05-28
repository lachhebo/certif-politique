# from unittest.mock import patch
from unittest import TestCase
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from certifia.feature_engineering import FeatureEngineering
from certifia.utils.multi_column_label_encode import MultiColumnLabelEncoder

TESTED_MODULE = 'certifia.utils.multi_column_label_encode'


class TestStringMethods(TestCase):
    def test_instantiate_feature_engineering_should_return_the_encoder(self):
        # given
        # when
        result = FeatureEngineering()
        # then
        self.assertEqual(result.training_columns, None)
        self.assertEqual(result.columns_to_dummify, None)
        self.assertIsInstance(result.label_encoder, MultiColumnLabelEncoder)

    def test_split_feature_label_shoud_return_a_df_of_feature_and_a_df_of_labels(self):
        # given
        training_columns = ["Col_Train_1", "Col_Train_2"]
        columns = ["Col_Train_1", "Col_Train_2", "RETARD A L'ARRIVEE"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A', 'A'], ['B', 'B', 'B'], ['A', 'B', 'C']])

        expected_df_X = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        expected_df_y = pd.Series(name="RETARD A L'ARRIVEE", data=['A', 'B', 'C'])

        # when
        result_X, result_y = FeatureEngineering(training_columns).split_feature_label(df)

        # then
        assert_frame_equal(expected_df_X, result_X)
        assert_series_equal(expected_df_y, result_y)

    def test_fit_transform_dummify_columns_should_return_a_dummified_df(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        columns_to_dummify = ["Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        # when
        result = FeatureEngineering(training_columns, columns_to_dummify).fit_transform_dummify_columns(df)
        # then
        # TODO
        print(result)

    def test_fit_transform_dummify_columns_should_not_return_dummified_df_when_no_columns_are_given(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        # when
        result = FeatureEngineering(training_columns).fit_transform_dummify_columns(df)
        # then
        # TODO
        print(result)

    def test_transform_dummify_columns_should_return_a_dummified_df(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        columns_to_dummify = ["Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        # when
        result = FeatureEngineering(training_columns, columns_to_dummify).transform_dummify_columns(df)
        # then
        # TODO
        print(result)

    def test_transform_dummify_columns_should_not_return_dummified_df_when_no_columns_are_given(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        # when
        result = FeatureEngineering(training_columns).transform_dummify_columns(df)
        # then
        # TODO
        print(result)
