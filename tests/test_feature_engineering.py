from datetime import date
from unittest.mock import patch
from unittest import TestCase

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from certifia.feature_engineering import FeatureEngineering
from certifia.utils.multi_column_label_encode import MultiColumnLabelEncoder

TESTED_MODULE = 'certifia.feature_engineering'


class TestFeatureEngineering(TestCase):
    def test_instantiate_feature_engineering_should_return_itself(self):
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
        columns = ["Col_Train_1", "Col_Train_2", "RETARD"]
        df = pd.DataFrame(columns=columns, data=[['A', 'A', 'A'], ['B', 'B', 'B'], ['A', 'B', 'C']])

        expected_df_X = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        expected_df_y = pd.Series(name="RETARD", data=['A', 'B', 'C'])

        # when
        result_X, result_y = FeatureEngineering(training_columns, label_name="RETARD").split_X_y(df)

        # then
        assert_frame_equal(expected_df_X, result_X)
        assert_series_equal(expected_df_y, result_y)

    @patch(f'{TESTED_MODULE}.MultiColumnLabelEncoder.transform')
    def test_fit_transform_dummify_columns_should_return_a_dummified_df(self, mock_fit_transform):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        expected_df = pd.DataFrame(columns=training_columns, data=[['A', 0], ['B', 1], ['A', 1]])
        mock_fit_transform.return_value = expected_df

        # when
        result = FeatureEngineering(training_columns, ["Dum"]).fit_transform_dummify_columns(df)

        # then
        assert_frame_equal(result, expected_df)

    def test_fit_transform_dummify_columns_should_not_return_dummified_df_when_no_columns_are_given(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])

        # when
        result = FeatureEngineering(training_columns).fit_transform_dummify_columns(df)

        # then
        assert_frame_equal(result, df)

    @patch(f'{TESTED_MODULE}.MultiColumnLabelEncoder.transform')
    def test_transform_dummify_columns_should_return_a_dummified_df(self, mock_transform):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])
        expected_df = pd.DataFrame(columns=training_columns, data=[['A', 0], ['B', 1], ['A', 1]])
        mock_transform.return_value = expected_df

        # when
        result = FeatureEngineering(training_columns, ["Dum"]).transform_dummify_columns(df)

        # then
        assert_frame_equal(result, expected_df)

    def test_transform_dummify_columns_should_not_return_dummified_df_when_no_columns_are_given(self):
        # given
        training_columns = ["Col_Train_1", "Dum"]
        df = pd.DataFrame(columns=training_columns, data=[['A', 'A'], ['B', 'B'], ['A', 'B']])

        # when
        result = FeatureEngineering(training_columns).transform_dummify_columns(df)

        # then
        assert_frame_equal(result, df)

    @patch('pandas.to_datetime')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.fit_transform_dummify_columns')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.split_X_y')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.cleaning')
    def test_fit_call_multiple_feature_engineering_method(
            self,
            mock_cleaning,
            mock_split_x_y,
            mock_fit_transform_dummify,
            mock_to_datetime
    ):
        # given
        training_columns = ["Col_Train_1", "Dum", "DATE"]
        df = pd.DataFrame(columns=training_columns)
        mock_split_x_y.return_value = pd.DataFrame, pd.DataFrame

        # when
        FeatureEngineering(training_columns).fit(df)

        # then
        self.assertEqual(mock_cleaning.call_count, 1)
        self.assertEqual(mock_split_x_y.call_count, 1)
        self.assertEqual(mock_fit_transform_dummify.call_count, 1)
        self.assertEqual(mock_to_datetime.call_count, 1)

    @patch('pandas.to_datetime')
    @patch(f'{TESTED_MODULE}.FeatureEngineering.transform_dummify_columns')
    def test_transform_should_return_dataframe_with_a_call_of_self_transform_dummify(
            self,
            mock_transform_dummi,
            mock_to_datetime
    ):
        # given
        training_columns = ["Col_Train_1", "Dum", "DATE", "MOIS", "SEMAINE"]
        df = pd.DataFrame(columns=["Col_Train_1", "Dum", "DATE"])
        mock_transform_dummi.return_value = df

        # when
        result = FeatureEngineering(training_columns).transform(df)

        # then
        self.assertEqual(mock_transform_dummi.call_count, 1)
        self.assertEqual(mock_to_datetime.call_count, 1)
        self.assertListEqual(result.columns.tolist(), training_columns)

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
        result = FeatureEngineering(
            training_columns,
            ["Dum"]
        ).get_average_plane_take_off_or_landing_by_day(df, 'AEROPORT')

        # then
        assert_series_equal(result, expected_df)

    # test d'intégration
    def test_fit_return_features_and_labels_dataframe(self):
        # given
        training_columns = ["Col_Train_1", "Dum", "DATE"]
        dummi_columns = ["Dum"]
        df = pd.DataFrame(
            columns=["Col_Train_1", "Dum", "DATE", "RETARD A L'ARRIVEE"],
            data=[['A', 'A', '01-01-2020', 1], ['B', 'B', '02-01-2020', 2], ['A', 'B', '03-01-2020', 3]]
        )
        expected_X = pd.DataFrame(columns=["Col_Train_1", "Dum", "DATE"], data=[
            ['A', 0, date(2020, 1, 1)],
            ['B', 1, date(2020, 2, 1)],
            ['A', 1, date(2020, 3, 1)]])
        expected_y = pd.Series(name="RETARD A L'ARRIVEE", data=[1, 2, 3])

        # when
        result_X, result_y = FeatureEngineering(training_columns, dummi_columns).fit(df)

        # then
        assert_frame_equal(result_X, expected_X, check_dtype=False)
        assert_series_equal(result_y, expected_y)

    def test_transform_return_features_dataframe(self):
        # given
        training_columns = ["Col_Train_1", "Dum", "DATE", "MOIS", "SEMAINE"]
        df = pd.DataFrame(
            columns=["Col_Train_1", "Dum", "DATE"],
            data=[['A', 'A', '01-01-2020'],
                  ['B', 'B', '01-02-2020'],
                  ['A', 'B', '01-03-2020']])
        expected_df = pd.DataFrame(
            columns=training_columns,
            data=[
                ['A', 'A', date(2020, 1, 1), 1, 1],
                ['B', 'B', date(2020, 1, 2), 1, 1],
                ['A', 'B', date(2020, 1, 3), 1, 1]])

        # when
        result = FeatureEngineering(training_columns).transform(df)

        # then
        assert_frame_equal(result, expected_df, check_dtype=False)

    def test_fit_transform_return_features_dataframe(self):
        # given
        columns_to_dummify = ["Dum"]
        train_df = pd.DataFrame(
            columns=["Col_Train_1", "Dum", "DATE", "RETARD A L'ARRIVEE"],
            data=[['A', 'A', '01-01-2020', 1],
                  ['B', 'B', '01-02-2020', 2],
                  ['C', 'C', '01-03-2020', 3]])
        training_columns = ["Col_Train_1", "Dum", "DATE"]
        test_df = pd.DataFrame(
            columns=training_columns,
            data=[['C', 'A', '01-01-2020'],
                  ['A', 'B', '01-01-2020'],
                  ['B', 'C', '01-01-2020']])
        expected_df = pd.DataFrame(
            columns=training_columns,
            data=[['C', 0, date(2020, 1, 1)],
                  ['A', 1, date(2020, 1, 1)],
                  ['B', 2, date(2020, 1, 1)]])
        feature_engineering = FeatureEngineering(training_columns, columns_to_dummify)

        # when
        feature_engineering.fit(train_df)
        result = feature_engineering.transform(test_df)

        # then
        assert_frame_equal(result, expected_df, check_dtype=False)
