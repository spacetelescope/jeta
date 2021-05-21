import os
import tempfile
from time import sleep

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

        tmp = tempfile.TemporaryDirectory()
        with tmp as tempdir:
            os.environ['TELEMETRY_ARCHIVE'] = tempdir

            test_msid_list = ['TEST_MSID']
            expected_result = len(test_msid_list)
            with open(os.path.join(tempdir, 'colnames.pickle'), '+wb') as tmp_pickle:
                pickle.dump(test_msid_list, tmp_pickle)

            from jeta.archive.status import get_msid_count
            actual_result = get_msid_count()
            tmp.cleanup()

            assert actual_result == expected_result

    def test_get_list_of_activities(self):
        self.fail('Write a test for: test_get_list_of_activities')

    def test_get_list_of_files_in_range(self):
        self.fail('Write a test for: test_get_list_of_files_in_range')

    # @mock.patch.dict(os.environ, {
    #         "TELEMETRY_ARCHIVE": FIXTURE_ARCHIVE
    #     }
    # )
    # def test_get_msid_names(self):
    #     import pickle

    #     tmp = tempfile.TemporaryDirectory()

    #     with tmp as tempdir:
    #         print(tempdir)
    #         os.environ['TELEMETRY_ARCHIVE'] = tempdir

    #         test_msid_list = ['TEST_MSID']
    #         expected_result = test_msid_list

    #         with open(os.path.join(tempdir, 'colnames.pickle'), '+wb') as tmp_pickle:
    #             pickle.dump(test_msid_list, tmp_pickle)

    #         from jeta.archive.status import get_msid_names
    #         actual_result = get_msid_names()
    #         assert actual_result == expected_result
