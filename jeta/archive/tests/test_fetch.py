# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division, absolute_import

from copy import deepcopy

import numpy as np
import pytest

from .. import fetch
from .. import fetch_eng
from astropy.time import Time

print(fetch.__file__)

DATES_EXPECT1 = np.array(['2008:291:23:59:58.987', '2008:291:23:59:59.244',
                          '2008:291:23:59:59.500', '2008:291:23:59:59.756',
                          '2008:297:00:00:00.121', '2008:297:00:00:00.378',
                          '2008:297:00:00:00.634'])

DATES_EXPECT2 = np.array(['2008:291:23:59:54.119', '2008:291:23:59:55.144',
                          '2008:291:23:59:56.169', '2008:291:23:59:57.194',
                          '2008:291:23:59:58.219', '2008:291:23:59:59.244',
                          '2008:297:00:00:00.890', '2008:297:00:00:01.915',
                          '2008:297:00:00:02.940', '2008:297:00:00:03.965'])

DATES_EXPECT3 = np.array(['2008:002:21:48:10.000', '2008:002:21:48:20.000',
                          '2008:002:21:48:30.000', '2008:002:21:48:40.000',
                          '2008:002:21:48:50.000', '2008:002:21:49:00.000',
                          '2008:002:21:49:10.000', '2008:002:21:49:20.000',
                          '2008:002:21:49:30.000', '2008:002:21:49:40.000',
                          '2008:002:21:49:50.000'])

BAD_TIMES = ['2008:292:00:00:00 2008:297:00:00:00',
             '2008:305:00:12:00 2008:305:00:12:03',
             '2010:101:00:01:12 2010:101:00:01:25']


def test_filter_bad_times_overlap():
    """
    OK to supply overlapping bad times
    """
    msid_bad_times_cache = deepcopy(fetch.msid_bad_times)
    dat = fetch.MSID('aogbias1', '2008:290', '2008:300', stat='daily')
    fetch.read_bad_times(['aogbias1 2008:292:00:00:00 2008:297:00:00:00'])
    fetch.read_bad_times(['aogbias1 2008:292:00:00:00 2008:297:00:00:00'])
    dat.filter_bad_times()
    fetch.msid_bad_times = msid_bad_times_cache

    # Test repr, len, and dtype attribute here where we have an MSID object handy
    assert repr(dat) == ('<MSID start=2008:290:12:00:00.000 stop=2008:300:12:00:00.000'
                         ' len=5 dtype=float32 unit=rad/s stat=daily>')
    assert dat.dtype.name == 'float32'
    assert len(dat) == 5


def test_filter_bad_times_list():
    dat = fetch.MSID('aogyrct1', '2008:291', '2008:298')
    # 2nd test of repr here where we have an MSID object handy
    assert repr(dat) == ('<MSID start=2008:291:12:00:00.000 '
                         'stop=2008:298:12:00:00.000 len=2360195 dtype=int16>')

    dat.filter_bad_times(table=BAD_TIMES)
    dates = Time(dat.times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)

    dat = fetch.Msid('aogyrct1', '2008:291', '2008:298')
    dat.filter_bad_times(table=BAD_TIMES)
    dates = Time(dat.times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)


def test_msidset_filter_bad_times_list():
    dat = fetch.MSIDset(['aogyrct1'], '2008:291', '2008:298')
    dat.filter_bad_times(table=BAD_TIMES)
    dates = Time(dat['aogyrct1'].times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)

    dat = fetch.Msidset(['aogyrct1'], '2008:291', '2008:298')
    dat.filter_bad_times(table=BAD_TIMES)
    dates = Time(dat['aogyrct1'].times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)


def test_filter_bad_times_default():
    """Test bad times that come from msid_bad_times.dat"""
    dat = fetch.MSID('aogbias1', '2008:291', '2008:298')
    dat.filter_bad_times()
    dates = Time(dat.times[42140:42150]).date
    assert np.all(dates == DATES_EXPECT2)


def test_filter_bad_times_list_copy():
    dat = fetch.MSID('aogyrct1', '2008:291', '2008:298')
    dat2 = dat.filter_bad_times(table=BAD_TIMES, copy=True)
    dates = Time(dat2.times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)
    assert len(dat.vals) != len(dat2.vals)

    dat = fetch.Msid('aogyrct1', '2008:291', '2008:298')
    dat2 = dat.filter_bad_times(table=BAD_TIMES, copy=True)
    dates = Time(dat2.times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)
    assert len(dat.vals) != len(dat2.vals)


def test_msidset_filter_bad_times_list_copy():
    dat = fetch.MSIDset(['aogyrct1'], '2008:291', '2008:298')
    dat2 = dat.filter_bad_times(table=BAD_TIMES, copy=True)
    dates = Time(dat2['aogyrct1'].times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)

    dat = fetch.Msidset(['aogyrct1'], '2008:291', '2008:298')
    dat2 = dat.filter_bad_times(table=BAD_TIMES, copy=True)
    dates = Time(dat2['aogyrct1'].times[168581:168588]).date
    assert np.all(dates == DATES_EXPECT1)


def test_filter_bad_times_default_copy():
    """Test bad times that come from msid_bad_times.dat"""
    dat = fetch.MSID('aogbias1', '2008:291', '2008:298')
    dat2 = dat.filter_bad_times(copy=True)
    dates = Time(dat2.times[42140:42150]).date
    assert np.all(dates == DATES_EXPECT2)


def test_interpolate():
    dat = fetch.MSIDset(['aoattqt1', 'aogyrct1', 'aopcadmd'],
                        '2008:002:21:48:00', '2008:002:21:50:00')
    dat.interpolate(10.0)

    assert np.allclose(dat['aoattqt1'].vals,
                       np.array([-0.33072645, -0.33072633, -0.33072634,
                                  -0.33072632, -0.33072705,
                                  -0.33073644, -0.33076456, -0.33081424,
                                  -0.33090285, -0.33088904,
                                  -0.33099454, -0.33128471]))

    assert np.all(dat['aopcadmd'].vals ==
                  np.array(['NPNT', 'NPNT', 'NPNT', 'NMAN', 'NMAN', 'NMAN',
                            'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN']))

    assert np.all(dat['aogyrct1'].vals ==
                  np.array([-23261, -22131, -21000, -19878, -18714, -17301,
                             -15379, -12674, -8914, -3398, 3514, 11486],
                           dtype=np.int16))


def test_interpolate_msid():
    start = '2008:002:21:48:00'
    stop = '2008:002:21:50:00'
    dat = fetch.MSID('aoattqt1', start, stop)
    dat.interpolate(10.0, start, stop)
    assert np.allclose(dat.vals,
                       np.array([-0.33072645, -0.33072633, -0.33072634,
                                  -0.33072632, -0.33072705,
                                  -0.33073644, -0.33076456, -0.33081424,
                                  -0.33090285, -0.33088904,
                                  -0.33099454, -0.33128471]))

    dat = fetch.MSID('aogyrct1', start, stop)
    dat.interpolate(10.0, start, stop)
    assert np.all(dat.vals ==
                  np.array([-23349, -22219, -21087, -19960, -18810, -17425,
                             -15551, -12919,
                             -9252, -3887, 2943, 10858],
                           dtype=np.int16))

    dat = fetch.MSID('aopcadmd', start, stop)
    dat.interpolate(10.0, start, stop)
    assert np.all(dat.vals ==
                  np.array(['NPNT', 'NPNT', 'NPNT', 'NMAN',
                            'NMAN', 'NMAN', 'NMAN', 'NMAN',
                            'NMAN', 'NMAN', 'NMAN', 'NMAN']))


def test_interpolate_times_raise():
    start = '2008:002:21:48:00'
    stop = '2008:002:21:50:00'
    dat = fetch.MSID('aoattqt1', start, stop)
    with pytest.raises(ValueError):
        dat.interpolate(10.0, times=[1, 2])


def test_interpolate_times():
    dat = fetch.MSIDset(['aoattqt1', 'aogyrct1', 'aopcadmd'],
                        '2008:002:21:48:00', '2008:002:21:50:00')
    dt = 10.0
    times = dat.tstart + np.arange((dat.tstop - dat.tstart) // dt + 3) * dt
    dat.interpolate(times=times)

    assert np.all(DateTime(dat.times).date == DATES_EXPECT3)

    assert np.allclose(dat['aoattqt1'].vals,
                       [-0.33072634, -0.33072637, -0.33072674, -0.33072665, -0.33073477,
                        -0.330761, -0.33080694, -0.33089434, -0.33089264, -0.33097442,
                        -0.33123678])

    assert np.all(dat['aopcadmd'].vals ==
                  ['NPNT', 'NPNT', 'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN',
                   'NMAN', 'NMAN', 'NMAN'])

    assert np.all(dat['aogyrct1'].vals ==
                  [-22247, -21117, -19988, -18839, -17468, -15605, -13000, -9360,
                    -4052,  2752, 10648])


def test_interpolate_msid_times():
    start = '2008:002:21:48:00'
    stop = '2008:002:21:50:00'
    dat = fetch.MSID('aoattqt1', start, stop)
    dt = 10.0
    times = dat.tstart + np.arange((dat.tstop - dat.tstart) // dt + 3) * dt
    dat.interpolate(times=times)
    assert np.allclose(dat.vals,
                       [-0.33072634, -0.33072637, -0.33072674, -0.33072665, -0.33073477,
                         -0.330761, -0.33080694, -0.33089434, -0.33089264, -0.33097442,
                         -0.33123678])

    assert np.all(DateTime(dat.times).date == DATES_EXPECT3)

    dat = fetch.MSID('aogyrct1', start, stop)
    dat.interpolate(times=times)
    assert np.all(dat.vals ==
                  [-22247, -21117, -19988, -18839, -17468, -15605, -13000, -9360,
                    -4052,  2752, 10648])

    assert np.all(DateTime(dat.times).date == DATES_EXPECT3)

    dat = fetch.MSID('aopcadmd', start, stop)
    dat.interpolate(times=times)
    assert np.all(dat.vals ==
                  ['NPNT', 'NPNT', 'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN', 'NMAN',
                   'NMAN', 'NMAN', 'NMAN'])

    assert np.all(Time(dat.times).date == DATES_EXPECT3)


def test_interpolate_time_precision():
    """
    Check that floating point error is < 0.01 msec over 100 days
    """
    dat = fetch.Msid('tephin', '2010:001', '2010:100')
    dt = 60.06
    times = dat.tstart + np.arange((dat.tstop - dat.tstart) // dt + 3) * dt

    dat.interpolate(60.06)  # Not exact binary float
    dt = dat.times[-1] - dat.times[0]
    dt_frac = dt * 100 - round(dt * 100)
    assert abs(dt_frac) > 0.001

    dat = fetch.Msid('tephin', '2010:001', '2010:100')
    dat.interpolate(times=times)
    dt = dat.times[-1] - dat.times[0]
    dt_frac = dt * 100 - round(dt * 100)
    assert abs(dt_frac) < 0.001


def _assert_msid_equal(msid1, msid2):
    for attr in ('tstart', 'tstop', 'datestart', 'datestop', 'units', 'unit', 'stat'):
        assert getattr(msid1, attr) == getattr(msid2, attr)
    assert np.all(msid1.times == msid2.times)
    assert np.all(msid1.vals == msid2.vals)
    assert msid1.__class__ is msid2.__class__


def test_msid_copy():
    for MsidClass in (fetch.Msid, fetch.MSID):
        msid1 = MsidClass('aogbias1', '2008:291', '2008:298')
        msid2 = msid1.copy()
        _assert_msid_equal(msid1, msid2)

    # Make sure msid data sets are independent
    msid2.filter_bad()
    assert len(msid1.vals) != len(msid2.vals)


def test_msidset_copy():
    for MsidsetClass in (fetch.MSIDset, fetch.Msidset):
        msidset1 = MsidsetClass(['aogbias1', 'aogbias2'], '2008:291', '2008:298')
        msidset2 = msidset1.copy()

        for attr in ('tstart', 'tstop', 'datestart', 'datestop'):
            assert getattr(msidset1, attr) == getattr(msidset2, attr)

        assert list(msidset1.keys()) == list(msidset2.keys())
        for name in msidset1.keys():
            _assert_msid_equal(msidset1[name], msidset2[name])


def test_MSIDset_interpolate_filtering():
    """
    Filtering and interpolation
    """
    # Bung up some data same as documentation example
    dat = fetch.MSIDset(['aosares1', 'pitch_fss'], '2010:001:00:00:00', '2010:001:00:00:20')
    dat['aosares1'].bads[2] = True
    dat['pitch_fss'].bads[6] = True

    # Uninterpolated
    assert np.sum(dat['aosares1'].bads == 1)
    assert np.sum(dat['pitch_fss'].bads == 1)
    assert len(dat['aosares1']) == 4
    assert len(dat['pitch_fss']) == 20

    # False, False
    dati = dat.interpolate(dt=0.5, filter_bad=False, bad_union=False, copy=True)

    assert np.sum(dati['aosares1'].bads) == 8
    assert np.sum(dati['pitch_fss'].bads) == 2
    assert len(dati['aosares1']) == 25
    assert len(dati['pitch_fss']) == 25

    # False, True
    dati = dat.interpolate(dt=0.5, filter_bad=False, bad_union=True, copy=True)

    assert np.sum(dati['aosares1'].bads) == 10  # same as below
    assert np.sum(dati['pitch_fss'].bads) == 10
    assert len(dati['aosares1']) == 25
    assert len(dati['pitch_fss']) == 25

    # True, False (default settings) returns all interpolated time samples
    dati = dat.interpolate(dt=0.5, filter_bad=True, bad_union=False, copy=True)

    assert np.sum(dati['aosares1'].bads) is None  # same as below
    assert np.sum(dati['pitch_fss'].bads) is None
    assert len(dati['aosares1']) == 25
    assert len(dati['pitch_fss']) == 25

    # True, True returns only good (in union sense) interpolated time samples
    # so 25 - 10 bad samples = 15
    dati = dat.interpolate(dt=0.5, filter_bad=True, bad_union=True, copy=True)

    assert np.sum(dati['aosares1'].bads) is None  # same as below
    assert np.sum(dati['pitch_fss'].bads) is None
    assert len(dati['aosares1']) == 15
    assert len(dati['pitch_fss']) == 15

    # Finally test that copy works (dat didn't change)
    assert np.sum(dat['aosares1'].bads) == 1
    assert np.sum(dat['pitch_fss'].bads) == 1
    assert len(dat['aosares1']) == 4
    assert len(dat['pitch_fss']) == 20


def test_1999_fetch():
    for start, stop in (('1999:363:00:00:00', '2000:005:00:00:00'),  # Covering deadband
                        ('1999:363:00:00:00', '2000:002:00:00:00'),  # Stop within deadband
                        ('2000:002:00:00:00', '2000:004:00:00:00'),  # Start within deadband
                        ('1999:360:00:00:00', '1999:361:00:00:00')):  # 1999 before deadband
        dat = fetch.MSID('aopcadmd', start, stop)
        assert np.allclose(np.diff(dat.times), 1.025, atol=0.01)

        dat = fetch.MSIDset(['aopcadmd', 'aoattqt1'], start, stop)
        assert np.allclose(np.diff(dat['aopcadmd'].times), 1.025, atol=0.01)
        assert np.allclose(np.diff(dat['aoattqt1'].times), 1.025, atol=0.01)


def test_intervals_fetch_unit():
    """
    Test that fetches with multiple intervals get the units right
    """
    dat = fetch_eng.Msid('tephin', [('1999:350', '1999:355'), ('2000:010', '2000:015')])
    assert np.allclose(np.mean(dat.vals), 41.713467)

    dat = fetch_eng.Msid('tephin', [('1999:350', '1999:355'), ('2000:010', '2000:015')],
                         stat='5min')
    assert np.allclose(np.mean(dat.vals), 40.290966)

    dat = fetch_eng.Msid('tephin', [('1999:350', '1999:355'), ('2000:010', '2000:015')],
                         stat='daily')
    assert np.allclose(np.mean(dat.vals), 40.303955)

    dat = fetch_eng.Msid('tephin', '1999:350', '2000:010')
    assert np.allclose(np.mean(dat.vals), 41.646729)


def test_ctu_dwell_telem():
    """
    Ensure that bad values are filtered appropriately for dwell mode telem.
    This
    """
    dat = fetch_eng.Msid('dwell01', '2015:294', '2015:295')
    assert np.all(dat.vals < 190)
    assert np.all(dat.vals > 150)

    dat = fetch_eng.Msid('airu1bt', '2015:294', '2015:295')
    assert np.all(dat.vals < -4.95)
    assert np.all(dat.vals > -5.05)


def test_nonexistent_msids():
    with pytest.raises(ValueError) as err:
        fetch.Msid('asdfasdfasdfasdf', '2015:001', '2015:002')
    assert "MSID 'asdfasdfasdfasdf' is not" in str(err.value)


def test_daily_state_bins():
    dat = fetch.Msid('aoacaseq', '2016:232', '2016:235', stat='daily')
    for attr, val in (('n_BRITs', [0, 136, 0]),
                      ('n_KALMs', [83994, 83812, 83996]),
                      ('n_AQXNs', [159, 240, 113]),
                      ('n_GUIDs', [140, 104, 184])):
        assert np.all(getattr(dat, attr) == val)

    dat = fetch.Msid('aoacaseq', '2016:234:12:00:00', '2016:234:12:30:00', stat='5min')
    assert np.all(dat.n_BRITs == [0, 0, 51, 17, 0, 0])
