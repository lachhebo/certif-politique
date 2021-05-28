from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from certifia.utils.logger import Logger

TESTED_MODULE = 'certifia.utils.logger'


class TestLogger(TestCase):
    @freeze_time("2012-01-14")
    @patch(f'{TESTED_MODULE}.logging.FileHandler')
    def test_set_config__save_logs_in_a_specific_file(self, mock_file_handler):
        # given
        logger = Logger()

        # when
        logger.set_config(Path('/root'))

        # then
        mock_file_handler.assert_called_with('/root/logs/2012-01-14.log')


    @pytest.mark.parametrize('method_name', ['debug', 'info', 'critical', 'warning'])
    @patch(f'{TESTED_MODULE}.logging.FileHandler')
    def test_set_config__use_good_logging_method(self, mock_file_handler, method_name):
        # given
        logger = Logger()

        # when
        with patch(f'{TESTED_MODULE}.logging.Logger.{method_name}'):
            method = getattr(logger, method_name)
            method('fake message')

            # then
            mocked_method = getattr(logger.logger, method_name)
            mocked_method.assert_called_once_with('fake message')
