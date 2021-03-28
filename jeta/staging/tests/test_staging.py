import os
import unittest
from unittest import mock


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
