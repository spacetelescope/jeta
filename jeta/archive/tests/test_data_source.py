# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division, absolute_import

import numpy as np
import pytest


from .. import fetch, fetch_sci, fetch_eng
fetch_cxc = fetch


date1 = '2016:001:00:00:00.1'
date2 = '2016:001:00:00:02.0'
date3 = '2016:001:00:00:05.0'
date4 = '2016:001:00:00:35.0'


try:
    import maude
    maude.get_msids(msids='ccsdsid', start=date1, stop=date2)
except Exception as err:
    HAS_MAUDE = False
else:
    HAS_MAUDE = True


def test_data_source():
    """Unit test of _DataSource class"""
    assert fetch.data_source.sources() == ('cxc',)

    # Context manager
    with fetch.data_source('maude'):
        assert fetch.data_source.sources() == ('maude',)
    assert fetch.data_source.sources() == ('cxc',)

    with fetch.data_source('cxc', 'maude'):
        assert fetch.data_source.sources() == ('cxc', 'maude',)
    assert fetch.data_source.sources() == ('cxc',)

    fetch.data_source.set('maude')
    assert fetch.data_source.sources() == ('maude',)
    fetch.data_source.set('cxc')
    assert fetch.data_source.sources() == ('cxc',)

    with pytest.raises(ValueError) as err:
        fetch.data_source.set('blah')
    assert 'not in allowed set' in str(err.value)

    with pytest.raises(ValueError) as err:
        with fetch.data_source():
            pass
    assert 'must select at least' in str(err.value)


def test_options():
    with fetch.data_source('cxc', 'maude allow_subset=False param=1'):
        assert fetch.data_source.options() == {'cxc': {},
                                               'maude': {'allow_subset': False,
                                                         'param': 1}
                                               }
        assert fetch.data_source.sources() == ('cxc', 'maude')


@pytest.mark.skipif("not HAS_MAUDE")
def test_maude_data_source():
    """Fetch data from MAUDE"""
    with fetch.data_source('cxc'):
        datc = fetch.Msid('aogyrct1', date1, date3)
        assert len(datc.vals) == 19
        assert datc.data_source == {'cxc': {'start': '2016:001:00:00:00.287',
                                            'stop': '2016:001:00:00:04.900'}}

    with fetch.data_source('maude'):
        datm = fetch.Msid('aogyrct1', date1, date3)
        assert np.all(datm.vals == datc.vals)
        assert not np.all(datm.times == datc.times)
        assert datm.data_source == {'maude': {'start': '2016:001:00:00:00.203',
                                              'stop': '2016:001:00:00:04.815',
                                              'flags': {'tolerance': False, 'subset': False}}}

    with fetch.data_source('cxc', 'maude', 'test-drop-half'):
        datcm = fetch.Msid('aogyrct1', date1, date3)
        assert np.all(datcm.vals == datc.vals)
        assert datcm.data_source == {'cxc': {'start': '2016:001:00:00:00.287',
                                             'stop': '2016:001:00:00:02.337'},
                                     'maude': {'start': '2016:001:00:00:02.509',
                                               'stop': '2016:001:00:00:04.815',
                                               'flags': {'tolerance': False, 'subset': False}}}


@pytest.mark.skipif("not HAS_MAUDE")
def test_maude_allow_subset_false():
    """Fetch data from MAUDE with allow_subset false"""
    with fetch.data_source('maude allow_subset=False'):
        datm = fetch.Msid('aogyrct1', '2016:001:00:00:00.1', '2016:001:12:00:00')
        ds = datm.data_source
        assert list(ds.keys()) == ['maude']
        assert ds['maude']['flags']['subset'] is False

    with fetch.data_source('cxc'):
        datc = fetch.Msid('aogyrct1', '2016:001:00:00:00.1', '2016:001:12:00:00')
        ds = datc.data_source
        assert list(ds.keys()) == ['cxc']

    assert len(datc.vals) == len(datm.vals)
    assert np.all(datc.vals == datm.vals)


@pytest.mark.skipif("not HAS_MAUDE")
def test_units_eng_to_other():
    for fetch_ in fetch_sci, fetch_eng, fetch_cxc:
        dat1 = fetch_.Msid('tephin', date1, date4)
        with fetch_.data_source('maude'):
            dat2 = fetch_.Msid('tephin', date1, date4)

        assert 'cxc' in dat1.data_source
        assert 'maude' in dat2.data_source
        assert dat1.unit == dat2.unit
        assert np.allclose(dat1.vals, dat2.vals)


@pytest.mark.skipif("not HAS_MAUDE")
def test_msid_resolution():
    """
    Make sure that MSIDs that might be in one data source or the other
    are handled properly.
    """
    with fetch.data_source('cxc', 'maude', 'test-drop-half'):
        # dp_pitch only in CXC but this succeeds anyway
        dat = fetch.Msid('dp_pitch', date1, date4)
        assert list(dat.data_source.keys()) == ['cxc']
        print(len(dat.vals))

        # Not in either
        with pytest.raises(ValueError) as err:
            fetch.Msid('asdfasdfasdf', date1, date4)
        assert "MSID 'asdfasdfasdf' is not in CXC or MAUDE" in str(err.value)

        # ACIMG1D1 only in MAUDE
        dat = fetch.Msid('ACIMG1D1', date1, date4)
        assert list(dat.data_source.keys()) == ['maude']
        print(len(dat.vals))

    with fetch.data_source('cxc'):
        # In MAUDE but this is not selected
        with pytest.raises(ValueError):
            fetch.Msid('ACIMG1D1', date1, date4)


@pytest.mark.skipif("not HAS_MAUDE")
def test_no_stats_maude():
    """Cannot select stats='5min' or 'daily' with MAUDE data source"""
    with fetch.data_source('maude'):
        with pytest.raises(ValueError) as err:
            fetch.Msid('TEPHIN', date1, date4, stat='5min')
        assert 'MAUDE data source does not support' in str(err.value)


def test_zero_length_fetch_cxc():
    """Zero length fetch gives a data source with empty dict"""
    with fetch.data_source('cxc'):
        dat = fetch.Msid('tephin', '2015:001', '2015:001')
    assert dat.data_source == {'cxc': {}}


@pytest.mark.skipif("not HAS_MAUDE")
def test_zero_length_fetch_maude():
    """Zero length fetch gives a data source with empty dict"""
    with fetch.data_source('maude'):
        dat = fetch.Msid('tephin', '2015:001', '2015:001')
    assert dat.data_source == {'maude': {'flags': {'tolerance': False, 'subset': False}}}
