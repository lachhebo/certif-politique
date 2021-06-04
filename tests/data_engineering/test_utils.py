from unittest.mock import patch, mock_open

from app.data_engineering.utils import load_feature_engineering


@patch('pickle.load')
def test_load_feature_engineering(mock_pickle):
    # Given
    with patch('builtins.open', mock_open()) as mock_open_method:
        # When
        load_feature_engineering()

    # Then
    assert mock_open_method.call_count == 1
    assert mock_pickle.call_count == 1
