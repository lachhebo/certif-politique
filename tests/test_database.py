from unittest import TestCase
from unittest.mock import patch

from certifia.data_engineering.database import Batch1DB, Batch2DB, TestDB

TESTED_MODULE = 'certifia.data_engineering.database'


class TestDatabase(TestCase):
    @patch(f'{TESTED_MODULE}.sqlite3.connect')
    def test_batch1_database_setup_connection_on_context_manager(self, mock_connect):
        # given
        batch1 = Batch1DB()
        batch1_path = 'fake_path'
        batch1.batch1_path = 'fake_path'

        # when

        with batch1:
            pass

        # then
        mock_connect.assert_called_with(batch1_path)

    @patch(f'{TESTED_MODULE}.sqlite3.connect')
    def test_batch2_database_setup_connection_on_context_manager(self, mock_connect):
        # given
        batch2 = Batch2DB()
        batch2_path = 'fake_path'
        batch2.batch2_path = 'fake_path'

        # when

        with batch2:
            pass

        # then
        mock_connect.assert_called_with(batch2_path)

    @patch(f'{TESTED_MODULE}.sqlite3.connect')
    def test_testdb_database_setup_connection_on_context_manager(self, mock_connect):
        # given
        testdb = TestDB()
        testdb_path = 'fake_path'
        testdb.test_path = 'fake_path'

        # when

        with testdb:
            pass

        # then
        mock_connect.assert_called_with(testdb_path)
