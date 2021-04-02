# base modules
import os
import time
from pathlib import Path

# testing, mocks, and fixtures modules
import unittest
import pytest
from unittest.mock import patch

# jeta specific modules
from jeta.archive.utils import get_env_variable

from jeta.staging.manage import (
    get_staged_files_by_date,
    remove_activity,
    _format_activity_destination,
    _create_activity_staging_area
)

HOME = str(Path.home())
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


class TestStaging(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_smoke_test(self):
        assert 1 == 1

    def test_can_import_manage(self):
        import importlib
        result = importlib.find_loader('jeta.staging.manage.py')

        assert result is not None

    def test_get_files_by_date(self):

        result = get_staged_files_by_date(0, time.time())

        assert len(result) == 1

    @patch('jeta.staging.manage.os.path.exists')
    @patch('jeta.staging.manage.os.mkdir')
    def test_create_activity_call_mkdir(self, mock_mkdir, mock_exists):
        mock_exists.return_value = False

        _activity = 'TEST_ACTIVITY'
        expected_path = f"{STAGING_DIRECTORY}{'TEST_ACTIVITY'}"

        _create_activity_staging_area(_activity)

        mock_mkdir.assert_called_once_with(expected_path)

    @patch('jeta.staging.manage.os.path.exists')
    def test_create_activity_already_exists_raise_error(self, mock_exists):

        with pytest.raises(IOError):
            mock_exists.return_value = True

            _activity = 'TEST_ACTIVITY'

            _create_activity_staging_area(_activity)

    @patch('jeta.staging.manage.os.path.exists')
    @patch('jeta.staging.manage.os.remove')
    def test_remove_activity(self, mock_remove, mock_exists):

        mock_exists.return_value = True

        _activity = 'TEST_ACTIVITY'
        expected_path = f"{STAGING_DIRECTORY}{'TEST_ACTIVITY'}"

        remove_activity(_activity)

        mock_remove.assert_called_once_with(expected_path)

    def test_add_activity(self):
        pytest.fail('No test implemented')

    def test_add_ingest_file_to_activity(self):
        pytest.fail('No test implemented')

    def test_get_files_for_activity(self):
        pytest.fail('No test implemented')

    def restore_activity_to_staging(self):
        pytest.fail('No test implemented')

    def test_format_activity_destination(self):
        _activity = "TEST_ACTIVITY"

        expected = f'{STAGING_DIRECTORY}{_activity}/'
        actual = _format_activity_destination(_activity)
        print(actual)

        assert expected == actual
