import pandas as pd
from pandas.testing import assert_frame_equal

from app.data_engineering.data_cleaning import DataCleaning


def test_cleaning_remove_unused_columns():
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
    output_df = data_cleaning.cleaning(df)

    # then
    assert_frame_equal(expected_df.reset_index(drop=True), output_df.reset_index(drop=True), check_dtype=False)
