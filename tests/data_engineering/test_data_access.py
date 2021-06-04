from unittest import TestCase
from unittest.mock import Mock, call, patch

from app.data_engineering.data_access import read_db

TESTED_MODULE = 'app.data_engineering.data_access'


class TestDataAccess(TestCase):
    @patch('sqlite3.connect')
    @patch(f'{TESTED_MODULE}.pandas.read_sql_query')
    def test_batch1_database_setup_connection_on_context_manager(self, mock_read_pandas, mock_connect):
        # given
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        calls = [
            call('SELECT * from vols', mock_conn),
            call('SELECT * from aeroports', mock_conn),
            call('SELECT * from compagnies', mock_conn),
            call('SELECT * from prix_fuel', mock_conn),
        ]

        # when

        read_db()

        # then
        mock_read_pandas.assert_has_calls(calls, any_order=True)
