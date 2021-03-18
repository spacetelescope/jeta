import os
import tempfile

import unittest
from unittest import mock


JETA_TEST_CACHE = '/Users/dkauffman/jeta-test-cache'
FIXTURE_ARCHIVE = '/Users/dkauffman/jeta-test-cache/archive'
FIXTURE_STAGE = '/Users/dkauffman/jeta-test-cache/stage'


class TestArchiveStatusAPI(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_smoke_test(self):
        assert 1 == 1

    @mock.patch.dict(os.environ, {
            "TELEMETRY_ARCHIVE": FIXTURE_ARCHIVE
        }
    )
    def test_get_msid_count(self):
        import pickle

        with tempfile.TemporaryDirectory() as tempdir:
            os.environ['TELEMETRY_ARCHIVE'] = tempdir

            test_msid_list = ['TEST_MSID']
            expected_result = len(test_msid_list)

            pickle.dump(test_msid_list, open(
                os.path.join(tempdir, 'colnames.pickle') , '+wb')
            )

            from jeta.archive.status import get_msid_count
            count = get_msid_count()
            assert count == expected_result

# class TestStagingStatusAPI(unittest.TestCase):
