#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" Retrieve data from the archive using jeta.archive.fetch module.
"""
from __future__ import print_function, division, absolute_import


import os
import sys
import time
import six
import pickle

import contextlib

import logging
import operator

import fnmatch
import collections
import warnings
import re
import json

import datetime
from datetime import timedelta

import numpy as np

from astropy.io import ascii
from astropy.time import Time

import h5py
import tables

import pyyaks.context

from jeta.archive import (
    cache,
    file_defs
)

from jeta.archive.units import Units


from jeta.archive.utils import get_env_variable
# from jeta.version import __version__, __git_version__

try:
    import fastss
except ImportError:
    pass

# Module-level units, defaults to CXC units (e.g. Kelvins etc)
UNITS = Units(system='cxc')

# Module-level control of whether MSID.fetch will cache the last 30 results
CACHE = False

ENG_ARCHIVE = get_env_variable('ENG_ARCHIVE')
JETA_SCRIPTS = get_env_variable('JETA_SCRIPTS')
ALL_KNOWN_MSID_SEMANTICS_H5 = get_env_variable('ALL_KNOWN_MSID_METAFILE')

IGNORE_COLNAMES = (
        'TIME',
        'MJF',
        'MNF',
        'TLM_FMT',
        'ENGINEERINGNUMERICVALUE',
        'ID',
        'OBSERVATORYTIME',
        'GROUNDTIME',
        'APID',
        'ENGINEERINGTEXTVALUE',
        'ALARMSTATUS'
        'id',
        'observatoryTime',
        'groundTime',
        'apid',
        'engineeringNumericValue',
        'engineeringTextValue',
        'alarmStatus',
)

# Dates near the start of 2000 that demarcates the split between the 1999 data
# and post-2000 data.  The 1999 data goes out to at least 2000:005:13:00:00,
# while post-2000 data starts as late as 2000:001:11:58:59.  Dates between LO
# and HI get taken from either 1999 or post-2000.  The times are 4 millisec before
# a minor frame boundary to avoid collisions.
DATE2000_LO = Time('2000:001:00:00:00.090').yday
DATE2000_HI = Time('2000:003:00:00:00.234').yday

# Earliest possible date data can be retrieved from the archive
JETA_EPOCH = '2008:001:00:00:00.000'

# Latest possible date and time that data can be retrieved from the archive
tomorrow = datetime.datetime.now() + timedelta(days=1)
LATEST_TLM_DATE = f"{tomorrow.timetuple().tm_year}:{tomorrow.timetuple().tm_yday:03d}:00:00:00.000"

# Maximum number of MSIDs that should ever match an input MSID spec
# (to prevent accidentally selecting a very large number of MSIDs)
MAX_GLOB_MATCHES = 10

# Special-case state codes that override those in the TDB
STATE_CODES = {
               # SIMDIAG
               '3SDSWELF': [(0, 'F'), (1, 'T')],
               '3SDSYRS': [(0, 'F'), (1, 'T')],
               '3SDWMRS': [(0, 'F'), (1, 'T')],

               # SIM_MRG
               '3TSCMOVE': [(0, 'F'), (1, 'T')],
               '3FAMOVE': [(0, 'F'), (1, 'T')],
               '3SEAID': [(0, 'SEA-A'), (1, 'SEA-B')],
               '3SEARSET': [(0, 'F'), (1, 'T')],
               '3SEAROMF': [(0, 'F'), (1, 'T')],
               '3SEAINCM': [(0, 'F'), (1, 'T')],
               '3STAB2EN': [(0, 'DISABLE'), (1, 'ENABLE')],
               '3SMOTPEN': [(0, 'ENABLE'), (1, 'DISABLE')],
               '3SMOTSEL': [(0, 'TSC'), (1, 'FA')],
               '3SHTREN': [(0, 'DISABLE'), (1, 'ENABLE')],
               '3SEARAMF': [(0, 'F'), (1, 'T')],
               }

# Cached version (by content type) of first and last available times in archive
CONTENT_TIME_RANGES = {}

# Default source of data.
DEFAULT_DATA_SOURCE = 'jwst'


class _DataSource(object):
    """
    Context manager and quasi-singleton configuration object for managing the
    data_source(s) used for fetching telemetry.
    """
    _data_sources = (DEFAULT_DATA_SOURCE,)
    _allowed = ('jwst')

    def __init__(self, *data_sources):
        self._new_data_sources = data_sources

    def __enter__(self):
        self._orig_data_sources = self.__class__._data_sources
        self.set(*self._new_data_sources)

    def __exit__(self, type, value, traceback):
        self.__class__._data_sources = self._orig_data_sources

    @classmethod
    def set(cls, *data_sources):
        """
        Set current data sources.

        :param *data_sources: one or more sources (str)
        """
        if any(data_source.split()[0] not in cls._allowed for data_source in data_sources):
            raise ValueError('data_sources {} not in allowed set {}'
                             .format(data_sources, cls._allowed))

        if len(data_sources) == 0:
            raise ValueError('must select at least one data source in {}'
                             .format(cls._allowed))

        cls._data_sources = data_sources

    @classmethod
    def sources(cls, include_test=True):
        """
        Get tuple of current data sources names.

        :param include_test: include sources that start with 'test'
        :returns: tuple of data source names
        """
        if include_test:
            sources = cls._data_sources
        else:
            sources = [x for x in cls._data_sources if not x.startswith('test')]

        return tuple(source.split()[0] for source in sources)

    @classmethod
    def get_msids(cls, source):
        """
        Get the set of MSID names corresponding to ``source`` (e.g. 'cxc' or 'maude')

        :param source: str
        :returns: set of MSIDs
        """
        source = source.split()[0]

        if source == 'jwst':
            out = list(content.keys())
        else:
            raise ValueError('source must be jwst')

        return set(out)

    @classmethod
    def options(cls):
        """
        Get the data sources and corresponding options as a dict.

        Example::

          >>> data_source.set('jwst') or ('jwst allow_subset=False')
          >>> data_source.options()
          {'jwst': {}} or {'jwst': {'allow_subset': False}}

        :returns: dict of data source options
        """
        import ast

        out = {}
        for source in cls._data_sources:
            vals = source.split()
            name, opts = vals[0], vals[1:]
            out[name] = {}
            for opt in opts:
                key, val = opt.split('=')
                val = ast.literal_eval(val)
                out[name][key] = val

        return out

# Set up logging.
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logger = logging.getLogger('jeta.archive.fetch')
logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger.setFormatter(formatter)
logger.addHandler(NullHandler())
logger.propagate = False

# Public interface is a "data_source" module attribute
data_source = _DataSource


def search_both_sorted(a, v):
    """Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted float array `a` such that, if the
    corresponding elements in float array `v` were inserted before the indices,
    the order of `a` would be preserved.

    Similar to np.searchsorted but BOTH `a` and `v` must be sorted in ascending
    order.  If `len(v) < len(a) / 100` then the normal `np.searchsorted` is
    called.  Otherwise both `v` and `a` are cast to `np.float64` internally
    and a Cython function is called to compute the indices in a fast way.

    Parameters
    ----------
    :param a: input float array, sorted in ascending order
    :param v: float values to insert into `a`, sorted in ascending order

    :returns: indices as int np.array
    """
    # If `v` is not comparable in length to `a` then np.searchsorted is faster.
    # Also require that both inputs be floating type.
    if a.dtype.kind != 'f' or v.dtype.kind != 'f' or len(v) < len(a) // 100:
        return np.searchsorted(a, v)
    else:
        # _search_both_sorted requires float64.  If already float64 this
        # just returns a view of the original (taking no extra memory).
        v = np.asarray(v, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        return fastss._search_both_sorted(a, v)

def _interpolate_vectorized(yin, xin, xout, method='linear', sorted=False):
    """
    Interpolate the curve defined by (xin, yin) at points xout.  The array
    xin must be monotonically increasing.  The output has the same data type as
    the input yin.

    :param yin: y values of input curve
    :param xin: x values of input curve
    :param xout: x values of output interpolated curve
    :param method: interpolation method ('linear' | 'nearest')
    :param sorted: `xout` values are sorted so use `search_both_sorted`

    @:rtype: numpy array with interpolated curve
    """

    lenxin = len(xin)
    if sorted:
        i1 = search_both_sorted(xin, xout)
    else:
        i1 = np.searchsorted(xin, xout)

    i1[ i1==0 ] = 1
    i1[ i1==lenxin ] = lenxin-1

    x0 = xin[i1-1]
    x1 = xin[i1]
    y0 = yin[i1-1]
    y1 = yin[i1]

    if method == 'linear':
        yout = (xout - x0) / (x1 - x0) * (y1 - y0) + y0
    elif method == 'nearest':
        yout = np.where(np.abs(xout - x0) < np.abs(xout - x1), y0, y1)
    else:
        raise ValueError('Invalid interpolation method: %s' % method)

    return yout


def _interpolate_cython(yin, xin, xout, method='linear', sorted=True):
    """
    Interpolate the curve defined by (xin, yin) at points xout.  Both arrays
    xin and xout must be monotonically increasing.  The output has the same
    data type as the input yin.

    This uses the Cython version and is faster for len(xout) >~ len(xin) / 100.

    :param yin: y values of input curve
    :param xin: x values of input curve
    :param xout: x values of output interpolated curve
    :param method: interpolation method ('linear' | 'nearest')
    :param sorted: `xout` values are sorted (must be True)

    @:rtype: numpy array with interpolated curve
    """

    if not sorted:
        raise ValueError('Input arrays must be sorted')

    if xin.dtype.kind != 'f' or xout.dtype.kind != 'f':
        raise ValueError('Input arrays must both be float type')

    xin = np.asarray(xin, dtype=np.float64)
    xout = np.asarray(xout, dtype=np.float64)

    if method == 'nearest':
        idxs = fastss._nearest_index(xin, xout)
        return yin[idxs]
    elif method == 'linear':
        yin64 = np.asarray(yin, dtype=np.float64)
        yout = fastss._interp_linear(yin64, xin, xout)
        return np.asarray(yout, dtype=yin.dtype)
    else:
        raise ValueError('Invalid interpolation method: {}'.format(method))



def not_np_interpolate(yin, xin, xout, method='linear', sorted=False, cython=True):
    """
    Interpolate the curve defined by (xin, yin) at points xout.  The array
    xin must be monotonically increasing.  The output has the same data type as
    the input yin.

    :param yin: y values of input curve
    :param xin: x values of input curve
    :param xout: x values of output interpolated curve
    :param method: interpolation method ('linear' | 'nearest')
    :param sorted: `xout` values are sorted so use `search_both_sorted`
    :param cython: use Cython interpolation code if possible (default=True)

    @:rtype: numpy array with interpolated curve
    """

    if (cython and sorted and
        xin.dtype.kind == 'f' and xout.dtype.kind == 'f'
        and (yin.dtype.kind == 'f' or method == 'nearest')
        and len(xout) >= len(xin) // 100):
        func = _interpolate_cython
    else:
        func = _interpolate_vectorized

    return func(yin, xin, xout, method=method, sorted=sorted)

def _get_start_stop_dates(times):
    if len(times) == 0:
        return {}
    else:
        return {
            'start': Time(times[0]).yday,
            'stop': Time(times[-1]).yday
        }


content = collections.OrderedDict()

# # Function to load MSID names from the files
# def load_msid_names():
#     all_colnames = dict()
#     with h5py.File(ALL_KNOWN_MSID_SEMANTICS_H5, 'r') as f:
#         all_colnames = list(f.keys())
#     return all_colnames

# # Load the MSID names
# all_colnames = load_msid_names()

# Save the names
# for colnames in all_colnames:
#     content.update((x, k) for x in sorted(colnames)
#                    if x not in IGNORE_COLNAMES)

# Cache of the most-recently used TIME array and associated bad values mask.
# The key is (content_type, tstart, tstop).
times_cache = dict(key=None)

ft = pyyaks.context.ContextDict('ft')


def get_units():
    """Get the unit system currently being used for conversions.
    """
    return UNITS['system']


def set_units(unit_system):
    """Set the unit system used for output telemetry values.  The default
    is "cxc".  Allowed values for ``unit_system``  are:

    ====  ==============================================================
    cxc   FITS standard units used in CXC archive files (basically MKS)
    sci   Same as "cxc" but with temperatures in degC instead of Kelvins
    eng   OCC engineering units (TDB P009, e.g. degF, ft-lb-sec, PSI)
    ====  ==============================================================

    :param unit_system: system of units (cxc, sci, eng)
    """
    UNITS.set_units(unit_system)


def read_bad_times(table):
    """Include a list of bad times from ``table`` in the fetch module
    ``bad_times`` registry.  This routine can be called multiple times with
    different tables and the bad times will be appended to the registry.  The
    table can include any number of bad time interval specifications, one per
    line.  A bad time interval line has three columns separated by whitespace,
    e.g.::

      aogbias1  2008:292:00:00:00  2008:297:00:00:00

    The MSID name is not case sensitive and the time values can be in any
    ``DateTime`` format.  Blank lines and any line starting with the #
    character are ignored.
    """
    bad_times = ascii.read(table, format='no_header',
                           names=['msid', 'start', 'stop'])

    for msid, start, stop in bad_times:
        msid_bad_times.setdefault(msid.upper(), []).append((start, stop))

# Set up bad times dict
msid_bad_times = dict()
# read_bad_times(os.path.join(JETA_SCRIPTS, 'msid_bad_times.dat'))


def msid_glob(msid):

    """Get the archive MSIDs matching ``msid``.

    The function returns a tuple of (msids, MSIDs) where ``msids`` is a list of
    MSIDs that is all lower case and (where possible) matches the input
    ``msid``.  The output ``MSIDs`` is all upper case and corresponds to the
    exact MSID names stored in the archive HDF5 files.

    :param msid: input MSID glob
    :returns: tuple (msids, MSIDs)
    """
    msids = collections.OrderedDict()
    MSIDS = collections.OrderedDict()

    sources = data_source.sources(include_test=False)
    for source in sources:
        ms, MS = _msid_glob(msid, source)
        msids.update((m, None) for m in ms)
        MSIDS.update((m, None) for m in MS)

    # if not msids:
    #     raise ValueError('MSID {!r} is not in {} data source(s)'
    #                      .format(msid, ' or '.join(x.upper() for x in sources)))

    return list(msids), list(MSIDS)


def _msid_glob(msid, source):


    """Get the archive MSIDs matching ``msid``.

    The function returns a tuple of (msids, MSIDs) where ``msids`` is a list of
    MSIDs that is all lower case and (where possible) matches the input
    ``msid``.  The output ``MSIDs`` is all upper case and corresponds to the
    exact MSID names stored in the archive HDF5 files.

    :param msid: input MSID glob
    :returns: tuple (msids, MSIDs)
    """

    source_msids = data_source.get_msids(source)

    MSID = msid.upper()
    # First try MSID or DP_<MSID>.  If success then return the upper
    # case version and whatever the user supplied (could be any case).
    for match in (MSID, 'DP_' + MSID):
        if match in source_msids:
            return [msid], [match]

    # Next try as a file glob.  If there is a match then return a
    # list of matches, all lower case and all upper case.  Since the
    # input was a glob the returned msids are just lower case versions
    # of the matched upper case MSIDs.
    for match in (MSID, 'DP_' + MSID):
        matches = fnmatch.filter(source_msids, match)
        if matches:
            if len(matches) > MAX_GLOB_MATCHES:
                raise ValueError(
                    'MSID spec {} matches more than {} MSIDs.  '
                    'Refine the spec or increase fetch.MAX_GLOB_MATCHES'
                    .format(msid, MAX_GLOB_MATCHES))
            return [x.lower() for x in matches], matches

    # msid not found for this data source
    return [], []


def _get_table_intervals_as_list(table, check_overlaps=True):
    """
    Determine if the input ``table`` looks like a table of intervals.  This can either be
    a structured array / Table with datestart / datestop or tstart / tstop columns,
    OR a list of lists.

    If so, return a list of corresponding start/stop tuples, otherwise return None.

    If ``check_overlaps`` is True then a check is made to assure that the supplied
    intervals do not overlap.  This is needed when reading multiple intervals with
    a single call to fetch, but not for bad times filtering.
    """
    intervals = None

    if isinstance(table, (list, tuple)):
        try:
            intervals = [(Time(row[0]).secs, Time(row[1]).secs)
                         for row in table]
        except:
            pass
    else:
        for prefix in ('date', 't'):
            start = prefix + 'start'
            stop = prefix + 'stop'
            try:
                intervals = [(Time(row[start]).unix, Time(row[stop]).unix)
                             for row in table]
            except:
                pass
            else:
                break

    # Got an intervals list, now sort
    if check_overlaps and intervals is not None:

        intervals = sorted(intervals, key=lambda x: x[0])

        # Check for overlaps
        if any(i0[1] > i1[0] for i0, i1 in zip(intervals[:-1], intervals[1:])):
            raise ValueError('Input intervals overlap')

    return intervals


class MSID(object):
    """Fetch data from the engineering telemetry archive into an MSID object.

    The input ``msid`` is case-insensitive.

    :param msid: a mnemonic (case-insensitive) as a string.
    :param start: start date of telemetry, formatted YYYY:DDD (optional, default is beginning of archive)
    :param stop: stop date of telemetry, formatted YYYY:DDD (optional, default is current time)
    :param filter_bad: automatically filter out bad values (optional)
    :param stat: return statistics, '5min' or 'daily' (optional)

    :returns: MSID instance, ex. data = fetch.MSID(<MSID>, start='2021:293:12:00', stop='2021:293:14:00')


    """
    units = UNITS
    fetch = sys.modules[__name__]

    def __init__(self, msid, start=JETA_EPOCH, stop=LATEST_TLM_DATE, filter_bad=False, stat=None):

        self.msid = msid.lower()
        self.MSID = msid.upper()

        # Capture the current module units
        self.units = Units(self.units['system'])
        self.unit = self.units.get_msid_unit(self.MSID)
        self.stat = stat

        if stat:
            self.dt = {
                '5min': 328, 
                'daily': 86400
            }[stat]

        # If ``start`` is actually a table of intervals then fetch
        # each interval separately and concatenate the results
        intervals = _get_table_intervals_as_list(start, check_overlaps=True)
        if intervals is not None:
            start, stop = intervals[0][0], intervals[-1][1]

        self.tstart = Time(start, format='yday').unix
        self.tstop = (Time(stop, format='yday').unix if stop else
                      Time(Time.now()).unix) 

        self.datestart = Time(self.tstart, format='unix').yday
        self.datestop = Time(self.tstop, format='unix').yday
        
        self.data_source = {}

        self.content = content.get(self.MSID)
    
        if self.datestart < DATE2000_LO and self.datestop > DATE2000_HI:
            intervals = [(self.datestart, DATE2000_HI),
                         (DATE2000_HI, self.datestop)]

        # Get the times, values, bad values mask from the HDF5 files archive
        if intervals is None:
            self._get_data()
        else:
            self._get_data_over_intervals(intervals)

        # TODO: Revive this feature
        # If requested filter out bad values and set self.bad = None
        # if filter_bad:
        #     self.filter_bad()

    def __len__(self):
        return len(self.vals)

    @property
    def dtype(self):
        return self.vals.dtype

    def __repr__(self):
        attrs = [self.__class__.__name__]
        for name, val in (
            ('start', self.datestart),
            ('stop', self.datestop),
            ('len', len(self)),
            ('dtype', self.dtype.name),
            ('unit', self.unit),
            ('stat', self.stat)
        ):
            if val is not None:
                attrs.append('{}={}'.format(name, val))

        return '<' + ' '.join(attrs) + '>'

    def _get_data_over_intervals(self, intervals):
        """
        Fetch intervals separately and concatenate the results.
        """
        msids = []
        for start, stop in intervals:
            msids.append(self.fetch.MSID(self.msid, start, stop, filter_bad=False, stat=self.stat))

        # No bad values column for stat='5min' or 'daily', but still need this attribute.
        if self.stat:
            self.bads = None

        self.colnames = msids[0].colnames
        for attr in self.colnames:
            vals = np.concatenate([getattr(msid, attr) for msid in msids])
            setattr(self, attr, vals)

    def _get_data(self):
        """ Get data from the Eng archive
        """
        logger.info('Getting data for %s between %s to %s',
                    self.msid, self.datestart, self.datestop)

        # Avoid stomping on caller's filetype 'ft' values with _cache_ft()
        with _cache_ft():
            # ft['content'] = self.content
            # ft['msid'] = self.MSID
            if self.stat:
                # ft['interval'] = self.stat
                self._get_stat_data(self.MSID, self.stat)
            else:
                self.colnames = ['vals', 'times', 'bads']
                args = (self.content, self.tstart, self.tstop, self.MSID, self.units['system'])

                if ('jwst' in data_source.sources()):
                    get_msid_data = self._get_archived_data
                    self.vals, self.times, self.bads = get_msid_data(*args)
                # if 'test-drop-half' in data_source.sources() and hasattr(self, 'vals'):
                #     # For testing purposes drop half the data off the end.  This assumes another
                #     # data_source like 'cxc' has been selected.
                #     idx = len(self.vals) // 2
                #     self.vals = self.vals[:idx]
                #     self.times = self.times[:idx]
                #     self.bads = self.bads[:idx]
                #     # Following assumes only one prior data source but ok for controlled testing
                #     for source in self.data_source:
                #         self.data_source[source] = _get_start_stop_dates(self.times)

    def _get_stat_data(self, msid, interval):
        """Do the actual work of getting stats values for an MSID from HDF5
        files"""
        filename = f"{ENG_ARCHIVE}/archive/data/tlm/stats/{interval}/{str(msid).upper()}.h5"
        logger.info('Opening %s', filename)

        def get_stat_data_from_server(filename, dt, tstart, tstop):
            """ This nested function reads the HDF5 file and calculates the range to slice

            :param filename: HDF5 file containing the stats
            :param dt: 328 or 86400 for 5min or daily respectively
            :param tstart: start of range as unix
            :param tstop:  end of slice range as unix
            """
            # TODO: switch to use Astropy.Time
            logger.info(f"FETCH: Telemetry data range {tstart} to {tstop} from {filename}")
            print(filename)
            
            h5 = tables.open_file(filename)
            table = h5.root.data
            times = (table.col('index') + 0.5) * dt
            times = Time(times, format="unix").unix

            row0, row1 = np.searchsorted(times, [tstart, tstop])
            table_rows = table[row0:row1]  # returns np.ndarray (structured array)
            h5.close()
            return (times[row0:row1], table_rows, row0, row1)
        times, table_rows, row0, row1 = \
            get_stat_data_from_server(filename,
                                      self.dt, self.tstart, self.tstop)
        logger.info('Closed %s', filename)

        self.bads = None
        self.times = times
        self.colnames = ['times']
        for colname in table_rows.dtype.names:
            # Don't like the way columns were named in the stats tables.
            # Fix that here.
            colname_out = _plural(colname) if colname != 'n' else 'samples'

            if colname_out in ('vals', 'mins', 'maxes', 'means',
                               'p01s', 'p05s', 'p16s', 'p50s',
                               'p84s', 'p95s', 'p99s'):
                vals = self.units.convert(self.MSID, table_rows[colname])
            elif colname_out == 'stds':
                vals = self.units.convert(self.MSID, table_rows[colname],
                                          delta_val=True)
            else:
                vals = table_rows[colname]

            setattr(self, colname_out, vals)
            self.colnames.append(colname_out)

        # Redefine the 'vals' attribute to be 'means' if it exists.  This is a
        # more consistent use of the 'vals' attribute and there is little use
        # for the original sampled version.
        if hasattr(self, 'means'):
            # Create new attribute midvals and add as a column (fixes kadi#17)
            self.colnames.append('midvals')
            self.midvals = self.vals
            self.vals = self.means

        # Possibly convert vals to unicode for Python 3+.  If this MSID is a
        # state-valued MSID (with string value) then `vals` is the only possible
        # string attribute.  None of the others like mins/maxes etc will exist.
        if not six.PY2:
            for colname in self.colnames:
                vals = getattr(self, colname)
                if vals.dtype.kind == 'S':
                    setattr(self, colname, vals.astype('U'))

    @staticmethod
    def _get_archived_data(content, tstart, tstop, msid, unit_system):

        """ Get the data from the archive and bind it to the MSID object.

        Do the actual work of getting time and values for an MSID from HDF5
        files.

        Parameters
        ==========

        content : ???
        tstart : astropy.time.TimeUnix
            the start date and time of the data being requested.
        tstop : astropy.time.TimeUnix
            the stop date and time of the data being requested.
        msid : str
        unit_system : ???

        """
        ft['content'] = 'tlm'

        # Get references to the arhiv files for this MSID
        values_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/values.h5" 
        times_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/times.h5"
        index_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/index.h5" 

        # The archive will use JD for its calculations
        start_jd = Time(tstart, format='unix', scale="utc").jd
        stop_jd = Time(tstop, format='unix', scale="utc").jd

        # Assume the index.h5 file is a table (named epoch) with 'epoch' and 'index' columns
        h5 = tables.open_file(index_filepath, 'r')
        index = h5.root.epoch[:]  # read the whole thing into a numpy structured array
        h5.close()

        # Interval that starts *before* start_jd, making sure to not go below 0
        idx0 = max(np.searchsorted(index['epoch'], start_jd, side='right') - 1, 0)

        # Interval that starts *after* stop_jd
        idx1 = np.searchsorted(index['epoch'], stop_jd, side='right')

        if len(index) == idx1:
            h5 = tables.open_file(values_filepath, 'r')
            dt = np.dtype([ ('epoch','<f8'), ('index', '<u8')])
            last_idx = np.array([(0, len(h5.root.values[0:-1]) )], dtype=dt)
            index = np.append(index, last_idx)
            h5.close()

        else:
            index = index[idx0:idx1 + 1]  # The +1 is so that the idx1 record is included

        # Start and stop rows which are guaranteed to contain start, stop
        row0 = index['index'][0]
        row1 = index['index'][-1]

        # Get the values
        h5 = tables.open_file(values_filepath, 'r')
        vals = h5.root.values[row0:row1]
        h5.close()

        # Get the times (still an array of delta times now)
        h5 = tables.open_file(times_filepath, 'r')
        jds = h5.root.times[row0:row1]
        h5.close()

        # Make the final time array now
        # jds = np.zeros_like(dts)

        # Apply the delta times.  This is the meat of the computation and is really
        # just a few lines.
        # for index0, index1 in zip(index[:-1], index[1:]):
        #     r0 = index0['index'] - row0
        #     r1 = index1['index'] - row0

        #     jds[r0:r1] = index0['epoch'] + np.cumsum(dts[r0:r1])

        # Final time filtering for exact user interval
        idx0, idx1 = np.searchsorted(jds, [start_jd, stop_jd])

        times= jds[idx0:idx1]
        vals = vals[idx0:idx1]
        
        # Covert to a time the original code expected
        times = Time(times, format="jd").unix

        # TODO: Remome from here; add type check and conversion into the ingest process.
        # try:
        #     vals = np.float64(vals)
        # except Exception as e:
        #     pass

        # Currenly no concept of bads
        bads = None

        return (vals, times, bads)


    @property
    def state_codes(self):
        """List of state codes tuples (raw_count, state_code) for state-valued
        MSIDs
        """
        if self.vals.dtype.kind not in ('S', 'U'):
            self._state_codes = None

        if self.MSID in STATE_CODES:
            self._state_codes = STATE_CODES[self.MSID]

        if not hasattr(self, '_state_codes'):
            import Ska.tdb
            try:
                states = Ska.tdb.msids[self.MSID].Tsc
            except:
                self._state_codes = None
            else:
                if states is None or len(set(states['CALIBRATION_SET_NUM'])) != 1:
                    warnings.warn('MSID {} has string vals but no state codes '
                                  'or multiple calibration sets'.format(self.msid))
                    self._state_codes = None
                else:
                    states = np.sort(states.data, order='LOW_RAW_COUNT')
                    self._state_codes = [(state['LOW_RAW_COUNT'],
                                          state['STATE_CODE']) for state in states]
        return self._state_codes


    @property
    def raw_vals(self):
        """Raw counts corresponding to the string state-code values that are
        stored in ``self.vals``
        """
        # If this is not a string-type value then there are no raw values
        if self.vals.dtype.kind not in ('S', 'U') or self.state_codes is None:
            self._raw_vals = None

        if not hasattr(self, '_raw_vals'):
            self._raw_vals = np.zeros(len(self.vals), dtype='int8') - 1
            # CXC state code telem all has same length with trailing spaces
            # so find max length for formatting below.
            max_len = max(len(x[1]) for x in self.state_codes)
            fmtstr = '{:' + str(max_len) + 's}'
            for raw_val, state_code in self.state_codes:
                ok = self.vals == fmtstr.format(state_code)
                self._raw_vals[ok] = raw_val

        return self._raw_vals

    @property
    def tdb(self):
        """Access the Telemetry database entries for this MSID
        """
        import Ska.tdb
        return Ska.tdb.msids[self.MSID]


    def interpolate(self, dt=None, start=None, stop=None, times=None):
        """Perform nearest-neighbor interpolation of the MSID to the specified
        time sequence.

        The time sequence steps uniformly by ``dt`` seconds starting at the
        ``start`` time and ending at the ``stop`` time.  If not provided the
        times default to the first and last times for the MSID.

        The MSID ``times`` attribute is set to the common time sequence.  In
        addition a new attribute ``times0`` is defined that stores the nearest
        neighbor interpolated time, providing the *original* timestamps of each
        new interpolated value for that MSID.

        If ``times`` is provided then this gets used instead of the default linear
        progression from ``start`` and ``dt``.

        :param dt: time step (sec, default=328.0)
        :param start: start of interpolation period (DateTime format)
        :param stop: end of interpolation period (DateTime format)
        :param times: array of times for interpolation (default=None)
        """

        if times is not None:
            if any(kwarg is not None for kwarg in (dt, start, stop)):
                raise ValueError('If "times" keyword is set then "dt", "start", '
                                 'and "stop" cannot be set')
            # Use user-supplied times that are within the range of telemetry.
            ok = (times >= self.times[0]) & (times <= self.times[-1])
            times = times[ok]
        else:

            dt = 328.0 if dt is None else dt
            tstart = Time(start).unix if start else self.times[0]
            tstop = Time(stop).unix if stop else self.times[-1]

            # Legacy method for backward compatibility.  Note that the np.arange()
            # call accrues floating point error.
            tstart = max(tstart, self.times[0])
            tstop = min(tstop, self.times[-1])
            times = np.arange(tstart, tstop, dt)

        logger.info('Interpolating index for %s', self.msid)
        print(f'Interpolating index for {self.msid}')
        indexes = not_np_interpolate(np.arange(len(self.times)),
                                        self.times, times,
                                        method='nearest', sorted=True)
        logger.info('Slicing on indexes')
        for colname in self.colnames:
            colvals = getattr(self, colname)
            if colvals is not None:
                setattr(self, colname, colvals[indexes])

        # Make a new attribute times0 that stores the nearest neighbor
        # interpolated times.  Then set the MSID times to be the common
        # interpolation times.
        self.times0 = self.times
        self.times = times


    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


    def filter_bad(self, bads=None, copy=False):
        """Filter out any bad values.

        After applying this method the "bads" column will be set to None to
        indicate that there are no bad values.

        :param bads: Bad values mask.  If not supplied then self.bads is used.
        :param copy: return a copy of MSID object with bad values filtered
        """
        obj = self.copy() if copy else self

        # If bad mask is provided then override any existing bad mask for MSID
        if bads is not None:
            obj.bads = bads

        # Nothing to do if bads is None (i.e. bad values already filtered)
        if obj.bads is None:
            return

        if np.any(obj.bads):
            logger.info('Filtering bad values for %s', obj.msid)
            ok = ~obj.bads
            colnames = (x for x in obj.colnames if x != 'bads')
            for colname in colnames:
                setattr(obj, colname, getattr(obj, colname)[ok])

        obj.bads = None

        if copy:
            return obj


    def filter_bad_times(self, start=None, stop=None, table=None, copy=False):
        """Filter out intervals of bad data in the MSID object.

        There are three usage options:

        - Supply no arguments.  This will use the global list of bad times read
          in with fetch.read_bad_times().
        - Supply both ``start`` and ``stop`` values where each is a single
          value in a valid DateTime format.
        - Supply an ``table`` parameter in the form of a 2-column table of
          start and stop dates (space-delimited) or the name of a file with
          data in the same format.

        The ``table`` parameter must be supplied as a table or the name of a
        table file, for example::

          bad_times = ['2008:292:00:00:00 2008:297:00:00:00',
                       '2008:305:00:12:00 2008:305:00:12:03',
                       '2010:101:00:01:12 2010:101:00:01:25']
          msid.filter_bad_times(table=bad_times)
          msid.filter_bad_times(table='msid_bad_times.dat')

        :param start: Start of time interval to exclude (any DateTime format)
        :param stop: End of time interval to exclude (any DateTime format)
        :param table: Two-column table (start, stop) of bad time intervals
        :param copy: return a copy of MSID object with bad times filtered
        """
        if table is not None:
            bad_times = ascii.read(table, format='no_header',
                                   names=['start', 'stop'])
        elif start is None and stop is None:
            bad_times = []
            for msid_glob, times in msid_bad_times.items():
                if fnmatch.fnmatch(self.MSID, msid_glob):
                    bad_times.extend(times)
        elif start is None or stop is None:
            raise ValueError('filter_times requires either 2 args '
                             '(start, stop) or no args')
        else:
            bad_times = [(start, stop)]

        obj = self.copy() if copy else self
        obj._filter_times(bad_times, exclude=True)
        if copy:
            return obj


    def remove_intervals(self, intervals, copy=False):
        """
        Remove telemetry points that occur within the specified ``intervals``

        This method is the converse of select_intervals().

        The ``intervals`` argument can be either a list of (start, stop) tuples
        or an EventQuery object from kadi.

        If ``copy`` is set to True then a copy of the MSID object is made prior
        to removing intervals, and that copy is returned.  The default is to
        remove intervals in place.

        This example shows fetching the pitch component of the spacecraft rate.
        After examining the rates, the samples during maneuvers are then removed
        and the standard deviation is recomputed.  This filters out the large
        rates during maneuvers::

          >>> m = fetch.Msid(<msid>, '2011:001', '2011:002')
          >>> m.vals.mean() * 3600 * 180 / np.pi  # rate in arcsec/sec
          3.9969393528801782
          >>> figure(1)
          >>> m.plot(',')


        :param intervals: EventQuery or iterable (N x 2) with start, stop dates/times
        :param copy: return a copy of MSID object with intervals removed
        """
        obj = self.copy() if copy else self
        obj._filter_times(intervals, exclude=True)
        if copy:
            return obj


    def select_intervals(self, intervals, copy=False):
        """
        Select telemetry points that occur within the specified ``intervals``

        This method is the converse of remove_intervals().

        The ``intervals`` argument can be either a list of (start, stop) tuples
        or an EventQuery object from kadi.

        If ``copy`` is set to True then a copy of the MSID object is made prior
        to selecting intervals, and that copy is returned.  The default is to
        selecte intervals in place.

        This example shows fetching the pitch component of the spacecraft rate.
        After examining the rates, the samples during maneuvers are then selected
        and the mean is recomputed.  This highlights the large rates during
        maneuvers::

          >>> m = fetch.Msid(<msid>, '2011:001', '2011:002')
          >>> m.vals.mean() * 3600 * 180 / np.pi  # rate in arcsec/sec
          3.9969393528801782
          >>> figure(1)
          >>> m.plot(',')


        :param intervals: EventQuery or iterable (N x 2) with start, stop dates/times
        :param copy: return a copy of MSID object with intervals selected
        """
        obj = self.copy() if copy else self
        obj._filter_times(intervals, exclude=False)
        if copy:
            return obj


    def _filter_times(self, intervals, exclude=True):
        """
        Filter the times of self based on ``intervals``.

        :param intervals: iterable (N x 2) with tstart, tstop in seconds
        :param exclude: exclude intervals if True, else include intervals
        """
        # Make an initial acceptance mask.  If exclude is True then initially
        # all values are allowed (ok=True).  If exclude is False (i.e. only
        # include the interval times) then ok=False everywhere.
        ok = np.empty(len(self.times), dtype=bool)
        ok[:] = exclude

        # See if the input intervals is actually a table of intervals
        intervals_list = _get_table_intervals_as_list(intervals, check_overlaps=False)
        if intervals_list is not None:
            intervals = intervals_list

        # Check if this is an EventQuery.  Would rather not import EventQuery
        # because this is expensive (django), so just look at the names in
        # object MRO.
        if 'EventQuery' in (cls.__name__ for cls in intervals.__class__.__mro__):
            intervals = intervals.intervals(self.datestart, self.datestop)

        for fmt in ('unix', 'yday', 'datetime'):
            try:
                intervals = [(Time(start, format=fmt).unix, Time(stop, format=fmt).unix)
                             for start, stop in intervals]
            except:
                pass
            else:
                break

        for tstart, tstop in intervals:
            if tstart > tstop:
                raise ValueError("Start time %s must be less than stop time %s"
                                 % (tstart, tstop))

            if tstop < self.times[0] or tstart > self.times[-1]:
                continue

            # Find the indexes of bad data.  Using side=left,right respectively
            # will exclude points exactly equal to the bad_times values
            # (though in reality an exact tie is extremely unlikely).
            i0 = np.searchsorted(self.times, tstart, side='left')
            i1 = np.searchsorted(self.times, tstop, side='right')
            ok[i0:i1] = not exclude

        colnames = (x for x in self.colnames)
        for colname in colnames:
            attr = getattr(self, colname)
            if isinstance(attr, np.ndarray):
                setattr(self, colname, attr[ok])


    def write_zip(self, filename, append=False):
        """Write MSID to a zip file named ``filename``

        Within the zip archive the data for this MSID will be stored in csv
        format with the name <msid_name>.csv.

        :param filename: output zipfile name
        :param append: append to an existing zipfile
        """
        import zipfile

        colnames = self.colnames[:]
        if self.bads is None and 'bads' in colnames:
            colnames.remove('bads')

        if self.state_codes:
            colnames.append('raw_vals')

        # Indexes value is not interesting for output
        if 'indexes' in colnames:
            colnames.remove('indexes')

        colvals = tuple(getattr(self, x) for x in colnames)
        fmt = ",".join("%s" for x in colnames)

        f = zipfile.ZipFile(filename, ('a' if append
                                       and os.path.exists(filename)
                                       else 'w'))
        info = zipfile.ZipInfo(self.msid + '.csv')
        info.external_attr = 0o664 << 16  # Set permissions
        info.date_time = time.localtime()[:7]
        info.compress_type = zipfile.ZIP_DEFLATED
        f.writestr(info,
                   ",".join(colnames) + '\n' +
                   '\n'.join(fmt % x for x in zip(*colvals)) + '\n')
        f.close()


    def logical_intervals(self, op, val, complete_intervals=True, max_gap=None):
        """Determine contiguous intervals during which the logical comparison
        expression "MSID.vals op val" is True.  Allowed values for ``op``
        are::

          ==  !=  >  <  >=  <=

        If ``complete_intervals`` is True (default) then the intervals are guaranteed to
        be complete so that the all reported intervals had a transition before and after
        within the telemetry interval.

        If ``max_gap`` is specified then any time gaps longer than ``max_gap`` are
        filled with a fictitious False value to create an artificial interval
        boundary at ``max_gap / 2`` seconds from the nearest data value.

        Returns a structured array table with a row for each interval.
        Columns are:

        * datestart: date of interval start
        * datestop: date of interval stop
        * duration: duration of interval (sec)
        * tstart: time of interval start (unix)
        * tstop: time of interval stop (unix)

        Examples::

          >>> dat = fetch.MSID('aomanend', '2010:001', '2010:005')
          >>> manvs = dat.logical_intervals('==', 'NEND')

          >>> dat = fetch.MSID('61PSTS02', '1999:200', '2000:001')
          >>> safe_suns = dat.logical_intervals('==', 'SSM', complete_intervals=False, max_gap=66)

        :param op: logical operator, one of ==  !=  >  <  >=  <=
        :param val: comparison value
        :param complete_intervals: return only complete intervals (default=True)
        :param max_gap: max allowed gap between time stamps (sec, default=None)
        :returns: structured array table of intervals
        """
        from . import utils

        ops = {'==': operator.eq,
               '!=': operator.ne,
               '>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le}
        try:
            op = ops[op]
        except KeyError:
            raise ValueError('op = "{}" is not in allowed values: {}'
                             .format(op, sorted(ops.keys())))

        # Do local version of bad value filtering
        if self.bads is not None and np.any(self.bads):
            ok = ~self.bads
            vals = self.vals[ok]
            times = self.times[ok]
        else:
            vals = self.vals
            times = self.times

        bools = op(vals, val)
        return utils.logical_intervals(times, bools, complete_intervals, max_gap)


    def state_intervals(self):
        """Determine contiguous intervals during which the MSID value
        is unchanged.

        Returns a structured array table with a row for each interval.
        Columns are:

        * datestart: date of interval start
        * datestop: date of interval stop
        * duration: duration of interval (sec)
        * tstart: time of interval start (unix)
        * tstop: time of interval stop (unix)
        * val: MSID value during the interval

        Example::

          dat = fetch.MSID('cobsrqid', '2010:001', '2010:005')
          obsids = dat.state_intervals()

        :param val: state value for which intervals are returned.
        :returns: structured array table of intervals
        """
        from . import utils

        # Do local version of bad value filtering
        if self.bads is not None and np.any(self.bads):
            ok = ~self.bads
            vals = self.vals[ok]
            times = self.times[ok]
        else:
            vals = self.vals
            times = self.times

        if len(self.vals) < 2:
            raise ValueError('Filtered data length must be at least 2')

        return utils.state_intervals(times, vals)


    def iplot(self, fmt='-b', fmt_minmax='-c', **plot_kwargs):
        """Make an interactive plot for exploring the MSID data.

        This method opens a new plot figure (or clears the current figure) and
        plots the MSID ``vals`` versus ``times``.  This plot can be panned or
        zoomed arbitrarily and the data values will be fetched from the archive
        as needed.  Depending on the time scale, ``iplot`` displays either full
        resolution, 5-minute, or daily values.  For 5-minute and daily values
        the min and max values are also plotted.

        Once the plot is displayed and the window is selected by clicking in
        it, the following key commands are recognized::

          a: autoscale for full data range in x and y
          m: toggle plotting of min/max values
          p: pan at cursor x
          y: toggle autoscaling of y-axis
          z: zoom at cursor x
          ?: print help

        Example::

          dat = fetch.Msid('aoattqt1', '2011:001', '2012:001', stat='5min')
          dat.iplot()
          dat.iplot('.b', '.c', markersize=0.5)

        Caveat: the ``iplot()`` method is not meant for use within scripts, and
        may give unexpected results if used in combination with other plotting
        commands directed at the same plot figure.

        :param fmt: plot format for values (default="-b")
        :param fmt_minmax: plot format for mins and maxes (default="-c")
        :param plot_kwargs: additional plotting keyword args

        """

        from .plot import MsidPlot
        self._iplot = MsidPlot(self, fmt, fmt_minmax, **plot_kwargs)


    def json(self):

        minValue = min(self.vals)
        maxValue = max(self.vals)

        telemetry = {
            'mnenmonic': {
                'level': 1,
                'minValue': minValue,
                'maxValue': maxValue,
                'datetimes': self.times,
                'values': self.vals,
            }
        }

        return json.dumps(telemetry)


    def plot(self, *args, **kwargs):
        """Plot the MSID ``vals`` using Ska.Matplotlib.plot_cxctime()

        This is a convenience function for plotting the MSID values.  It
        is equivalent to::

          plot_cxctime(self.times, self.vals, *args, **kwargs)

        where ``*args`` are additional arguments and ``**kwargs`` are
        additional keyword arguments that are accepted by ``plot_cxctime()``.

        Example::

          dat = fetch.Msid('tephin', '2011:001', '2012:001', stat='5min')
          dat.plot('-r', linewidth=2)

        """

        import matplotlib.pyplot as plt
        from .plot import plot_cxctime
        vals = self.raw_vals if self.state_codes else self.vals

        plot_cxctime(self.times, vals, *args, state_codes=self.state_codes,
                     **kwargs)
        plt.margins(0.02, 0.05)


    def __len__(self):
        return len(self.times)


class MSIDset(collections.OrderedDict):
    """Fetch a set of MSIDs from the engineering telemetry archive.

    :param msids: list of MSID names (case-insensitive)
    :param start: start date of telemetry, formatted YYYY:DDD (optional, default is beginning of archive)
    :param stop: stop date of telemetry, formatted YYYY:DDD (optional, default is current time)
    :param filter_bad: automatically filter out bad values (optional)
    :param stat: return statistics, '5min' or 'daily' (optional)

    :returns: Dict-like object containing MSID instances keyed by MSID name
    """
    MSID = MSID

    def __init__(self, msids, start=JETA_EPOCH, stop=None, filter_bad=False, stat=None):
        super(MSIDset, self).__init__()

        intervals = _get_table_intervals_as_list(start, check_overlaps=True)
        if intervals is not None:
            start, stop = intervals[0][0], intervals[-1][1]

        self.tstart = Time(start, format='yday').unix
        self.tstop = (Time(stop, format='yday').unix if stop else Time(Time.now()).unix)
        self.datestart = Time(self.tstart, format="unix").yday
        self.datestop = Time(self.tstop, format="unix").yday

        # Input ``msids`` may contain globs, so expand each and add to new list
        # new_msids = []
        # for msid in msids:
        #     new_msids.extend(msid_glob(msid)[0])
        for msid in msids:
            if intervals is None:
                self[msid] = self.MSID(msid, self.datestart, self.datestop,
                                       filter_bad=False, stat=stat)
            else:
                self[msid] = self.MSID(msid, intervals, filter_bad=False, stat=stat)

        if filter_bad:
            self.filter_bad()

    def __deepcopy__(self, memo=None):
        out = self.__class__([], None)
        for attr in ('tstart', 'tstop', 'datestart', 'datestop'):
            setattr(out, attr, getattr(self, attr))
        for msid in self:
            out[msid] = self[msid].copy()

        return out

    def copy(self):
        return self.__deepcopy__()

    def filter_bad(self, copy=False):
        """Filter bad values for the MSID set.

        This function applies the union (logical-OR) of bad value masks for all
        MSIDs in the set with the same content type.  The result is that the
        filtered MSID samples are valid for *all* MSIDs within the
        content type and the arrays all match up.

        For example::

          msids = fetch.MSIDset(['aorate1', 'aorate2', 'aogyrct1', 'aogyrct2'],
                                '2009:001', '2009:002')
          msids.filter_bad()

        Since ``aorate1`` and ``aorate2`` both have content type of
        ``pcad3eng`` they will be filtered as a group and will remain with the
        same sampling.  This will allow something like::

          plot(msids['aorate1'].vals, msids['aorate2'].vals)

        Likewise the two gyro count MSIDs would be filtered as a group.  If
        this group-filtering is not the desired behavior one can always call
        the individual MSID.filter_bad() function for each MSID in the set::

          for msid in msids.values():
              msid.filter_bad()

        :param copy: return a copy of MSID object with intervals selected
        """
        obj = self.copy() if copy else self

        for content in set(x.content for x in obj.values()):
            bads = None

            msids = [x for x in obj.values()
                     if x.content == content and x.bads is not None]
            for msid in msids:
                if bads is None:
                    bads = msid.bads.copy()
                else:
                    bads |= msid.bads

            for msid in msids:
                msid.filter_bad(bads)

        if copy:
            return obj

    def filter_bad_times(self, start=None, stop=None, table=None, copy=False):
        """Filter out intervals of bad data in the MSIDset object.

        There are three usage options:

        - Supply no arguments.  This will use the global list of bad times read
          in with fetch.read_bad_times().
        - Supply both ``start`` and ``stop`` values where each is a single
          value in a valid DateTime format.
        - Supply an ``table`` parameter in the form of a 2-column table of
          start and stop dates (space-delimited) or the name of a file with
          data in the same format.

        The ``table`` parameter must be supplied as a table or the name of a
        table file, for example::

          msidset.filter_bad_times()
          bad_times = ['2008:292:00:00:00 2008:297:00:00:00',
                       '2008:305:00:12:00 2008:305:00:12:03',
                       '2010:101:00:01:12 2010:101:00:01:25']
          msidset.filter_bad_times(table=bad_times)
          msidset.filter_bad_times(table='msid_bad_times.dat')

        :param start: Start of time interval to exclude (any DateTime format)
        :param stop: End of time interval to exclude (any DateTime format)
        :param table: Two-column table (start, stop) of bad time intervals
        :param copy: return a copy of MSID object with intervals selected
        """
        obj = self.copy() if copy else self

        for msid in obj.values():
            msid.filter_bad_times(start, stop, table)

        if copy:
            return obj

    def interpolate(self, dt=None, start=None, stop=None, filter_bad=True, times=None,
                    bad_union=False, copy=False):
        """
        Perform nearest-neighbor interpolation of all MSID values in the set
        to a common time sequence.  The values are updated in-place.

        **Times**

        The time sequence steps uniformly by ``dt`` seconds starting at the
        ``start`` time and ending at the ``stop`` time.  If not provided the
        times default to the ``start`` and ``stop`` times for the MSID set.

        If ``times`` is provided then this gets used instead of the default linear
        progression from ``start`` and ``dt``.

        For each MSID in the set the ``times`` attribute is set to the common
        time sequence.  In addition a new attribute ``times0`` is defined that
        stores the nearest neighbor interpolated time, providing the *original*
        timestamps of each new interpolated value for that MSID.

        **Filtering and bad values**

        If ``filter_bad`` is True (default) then bad values are filtered from
        the interpolated MSID set.  There are two strategies for doing this:

        1) ``bad_union = False``

           Remove the bad values in each MSID prior to interpolating the set to
           a common time series.  This essentially says to use all the available
           data individually.  Each MSID has bad data filtered individually
           *before* interpolation so that the nearest neighbor interpolation only
           finds good data.  This strategy is done when ``filter_union = False``,
           which is the default setting.

        2) ``bad_union = True``

          Mark every MSID in the set as bad at the interpolated time if *any*
          of them are bad at that time.  This stricter version is required when it
          is important that the MSIDs be truly correlated in time.  For instance
          this is needed for attitude quaternions since all four values must be
          from the exact same telemetry sample.  If you are not sure, this is the
          safer option.

        :param dt: time step (sec, default=328.0)
        :param start: start of interpolation period (DateTime format)
        :param stop: end of interpolation period (DateTime format)
        :param filter_bad: filter bad values
        :param times: array of times for interpolation (default=None)
        :param bad_union: filter union of bad values after interpolating
        :param copy: return a new copy instead of in-place update (default=False)
        """

        obj = self.copy() if copy else self

        msids = list(obj.values())  # MSID objects in the MSIDset

        # Ensure that tstart / tstop is entirely within the range of available
        # data fetched from the archive.
        max_fetch_tstart = max(msid.times[0] for msid in msids)
        min_fetch_tstop = min(msid.times[-1] for msid in msids)

        if times is not None:
            if any(kwarg is not None for kwarg in (dt, start, stop)):
                raise ValueError('If "times" keyword is set then "dt", "start", '
                                 'and "stop" cannot be set')
            # Use user-supplied times that are within the range of telemetry.
            ok = (times >= max_fetch_tstart) & (times <= min_fetch_tstop)
            obj.times = times[ok]
        else:
            # Get the nominal tstart / tstop range
            dt = 328.0 if dt is None else dt

            tstart = Time(start).unix if start else Time(obj.tstart, format="unix").unix
            tstop = Time(stop).unix if stop else Time(obj.tstop, format="unix").unix

            tstart = max(tstart, max_fetch_tstart)
            tstop = min(tstop, min_fetch_tstop)

            obj.times = np.arange((tstop - tstart) // dt + 1) * dt + tstart

        for msid in msids:

            if filter_bad and not bad_union:
                msid.filter_bad()
            logger.info('Interpolating index for %s', msid.msid)
            indexes = not_np_interpolate(np.arange(len(msid.times)),
                                            msid.times, obj.times,
                                            method='nearest', sorted=True)
            logger.info('Slicing on indexes')
            for colname in msid.colnames:
                colvals = getattr(msid, colname)
                if colvals is not None:
                    setattr(msid, colname, colvals[indexes])

            # Make a new attribute times0 that stores the nearest neighbor
            # interpolated times.  Then set the MSID times to be the common
            # interpolation times.

            msid.times0 = msid.times
            msid.times = obj.times

        if bad_union:
            common_bads = np.zeros(len(obj.times), dtype=bool)
            for msid in msids:
                if msid.stat is None and msid.bads is None:
                    warnings.warn('WARNING: {!r} MSID has bad values already filtered.\n'
                                  'This prevents `filter_bad_union` from working as expected.\n'
                                  'Use MSIDset (not Msidset) with filter_bad=False.\n')
                if msid.bads is not None:  # 5min and daily stats have no bad values
                    common_bads |= msid.bads

            # Apply the common bads array and optional filter out these bad values
            for msid in msids:
                msid.bads = common_bads
                if filter_bad:
                    msid.filter_bad()

            # Filter MSIDset-level times attr to match invididual MSIDs if filter_bad is True
            if filter_bad:
                obj.times = obj.times[~common_bads]

        if copy:
            return obj

    def write_zip(self, filename):
        """Write MSIDset to a zip file named ``filename``

        Within the zip archive the data for each MSID in the set will be stored
        in csv format with the name <msid_name>.csv

        :param filename: output zipfile name
        """
        append = False
        for msid in self.values():
            msid.write_zip(filename, append=append)
            append = True


class Msid(MSID):
    """
    Fetch data from the engineering telemetry archive into an MSID object.
    Same as MSID class but with filter_bad=True by default.

    :param msid: name of MSID (case-insensitive)
    :param msid: a mnemonic (case-insensitive) as a string.
    :param start: start date of telemetry, formatted YYYY:DDD (optional, default is beginning of archive)
    :param stop: stop date of telemetry, formatted YYYY:DDD (optional, default is current time)
    :param filter_bad: automatically filter out bad values (optional)
    :param stat: return statistics, '5min' or 'daily' (optional)

    :returns: MSID instance, ex. data = fetch.MSID(<MSID>, start='2021:293:12:00', stop='2021:293:14:00')
    """
    units = UNITS

    def __init__(self, msid, start=JETA_EPOCH, stop=None, filter_bad=True, stat=None):
        super(Msid, self).__init__(msid, start=start, stop=stop,
                                   filter_bad=filter_bad, stat=stat)


class Msidset(MSIDset):
    """Fetch a set of MSIDs from the engineering telemetry archive.
    Same as MSIDset class but with filter_bad=True by default.

    :param msids: list of MSID names (case-insensitive)
    :param msids: list of MSID names (case-insensitive)
    :param start: start date of telemetry, formatted YYYY:DDD (optional, default is beginning of archive)
    :param stop: stop date of telemetry, formatted YYYY:DDD (optional, default is current time)
    :param filter_bad: automatically filter out bad values (optional)
    :param stat: return statistics, '5min' or 'daily' (optional)

    :returns: Dict-like object containing MSID instances keyed by MSID name
    """
    MSID = MSID

    def __init__(self, msids, start=JETA_EPOCH, stop=None, filter_bad=True, stat=None):
        super(Msidset, self).__init__(msids, start=start, stop=stop,
                                      filter_bad=filter_bad, stat=stat)


class HrcSsMsid(Msid):
    """
    Fetch data from the engineering telemetry archive into an MSID object.
    Same as MSID class but with filter_bad=True by default.

    :param msid: name of MSID (case-insensitive)
    :param start: start date of telemetry (Chandra.Time compatible)
    :param stop: stop date of telemetry (current time if not supplied)
    :param filter_bad: automatically filter out bad values
    :param stat: return 5-minute or daily statistics ('5min' or 'daily')
    :param unit_system: Unit system (cxc|eng|sci, default=current units)

    :returns: MSID instance

    """
    units = UNITS

    def __new__(self, msid, start=JETA_EPOCH, stop=None, stat=None):
        ss_msids = '2TLEV1RT 2VLEV1RT 2SHEV1RT 2TLEV2RT 2VLEV2RT 2SHEV2RT'
        if msid.upper() not in ss_msids.split():
            raise ValueError('MSID {} is not in HRC secondary science ({})'
                             .format(msid, ss_msids))

        # If this is not full-resolution then add boolean bads mask to individual MSIDs
        msids = [msid, 'HRC_SS_HK_BAD']
        out = MSIDset(msids, start=start, stop=stop, stat=stat)
        if stat is not None:
            for m in msids:
                out[m].bads = np.zeros(len(out[m].vals), dtype=np.bool)

        # Set bad mask
        i_bads = np.flatnonzero(out['HRC_SS_HK_BAD'].vals > 0)
        out['HRC_SS_HK_BAD'].bads[i_bads] = True

        # For full-resolution smear the bad mask out by +/- 5 samples
        if stat is None:
            for i_bad in i_bads:
                i0 = max(0, i_bad - 5)
                i1 = i_bad + 5
                out['HRC_SS_HK_BAD'].bads[i0:i1] = True

        # Finally interpolate and filter out bad values
        out.interpolate(times=out[msid].times, bad_union=True, filter_bad=True)
        return out[msid]


class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            self.cache[args] = value = self.func(*args)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__


def get_time_range(msid, format=None):

    """
    Get the time range for the given ``msid``.

    :param msid: MSID name
    :param format: Output format (DateTime format, e.g. 'secs', 'date', 'greta')
    :returns: (tstart, tstop) in seconds
    """

    MSID = msid.upper()
    with _cache_ft():
        import tables

        content[msid] = ""
        ft['content'] = 'tlm'
        ft['msid'] = msid

        times_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/times.h5"
        # index_filepath = msid_files['mnemonic_index'].abs

        logger.info('Reading %s', times_filepath)
       
        times_h5 = tables.open_file(times_filepath)
        # index_h5 = tables.open_file(index_filepath)

        # sp_idx = int(index_h5.root.epoch[-1][1]) - 1

        tstart = times_h5.root.times[0] # index_h5.root.epoch[0][0] + np.cumsum(times_h5.root.times[0])[0]

        try:
            tstop = times_h5.root.times[-1] # index_h5.root.epoch[-1][0] + np.cumsum(times_h5.root.times[sp_idx:-1])[-1]
        except Exception as err:
            tstop = tstart
            #raise

        # index_h5.close()
        times_h5.close()

        if format == 'iso':
            tstart = Time(tstart, format='jd').iso
            if tstop is not None:
                tstop = Time(tstop, format='jd').iso

        if format == 'date':
            tstart = Time(tstart, format='jd').yday
            if tstop is not None:
                tstop = Time(tstop, format='jd').yday

        return tstart, tstop


def get_telem(msids, start=None, stop=None, sampling='full', unit_system='eng',
              interpolate_dt=None, remove_events=None, select_events=None,
              time_format=None, outfile=None, quiet=False,
              max_fetch_Mb=1000, max_output_Mb=100):
    """
    High-level routine to get telemetry for one or more MSIDs and perform
    common processing functions:

      - Fetch a set of MSIDs over a time range, specifying the sampling as
        either full-resolution, 5-minute, or daily data.
      - Filter out bad or missing data.
      - Interpolate (resample) all MSID values to a common uniformly-spaced time sequence.
      - Remove or select time intervals corresponding to specified Kadi event types.
      - Change the time format from seconds (seconds since 1998.0) to something more
        convenient like GRETA time.
      - Write the MSID telemetry data to a zipfile.

    :param msids: MSID(s) to fetch (string or list of strings)')
    :param start: Start time for data fetch (default=<stop> - 30 days)
    :param stop: Stop time for data fetch (default=NOW)
    :param sampling: Data sampling (full | 5min | daily) (default=full)
    :param unit_system: Unit system for data (eng | sci | cxc) (default=eng)
    :param interpolate_dt: Interpolate to uniform time steps (secs, default=None)
    :param remove_events: Remove kadi events expression (default=None)
    :param select_events: Select kadi events expression (default=None)
    :param time_format: Output time format (secs|date|greta|jd|..., default=secs)
    :param outfile: Output file name (default=None)
    :param quiet: Suppress run-time logging output (default=False)
    :param max_fetch_Mb: Max allowed memory (Mb) for fetching (default=1000)
    :param max_output_Mb: Max allowed memory (Mb) for file output (default=100)

    :returns: MSIDset object
    """
    from .get_telem import get_telem
    return get_telem(msids, start, stop, sampling, unit_system,
                     interpolate_dt, remove_events, select_events,
                     time_format, outfile, quiet,
                     max_fetch_Mb, max_output_Mb)


# @memoized
# def get_interval(content, tstart, tstop):
#     """
#     Get the approximate row intervals that enclose the specified ``tstart`` and
#     ``tstop`` times for the ``content`` type.

#     :param content: content type (e.g. 'pcad3eng', 'thm1eng')
#     :param tstart: start time (seconds)
#     :param tstop: stop time (seconds)

#     :returns: rowslice
#     """

#     ft['content'] = content


#     def get_interval_from_db(tstart, tstop, server):

#         import Ska.DBI

#         db = Ska.DBI.DBI(dbi='sqlite', server=os.path.join(*server))

#         query_row = db.fetchone('SELECT tstart, rowstart FROM archfiles '
#                                 'WHERE filetime < ? order by filetime desc',
#                                 (tstart,))
#         if not query_row:
#             query_row = db.fetchone('SELECT tstart, rowstart FROM archfiles '
#                                     'order by filetime asc')

#         rowstart = query_row['rowstart']

#         query_row = db.fetchone('SELECT tstop, rowstop FROM archfiles '
#                                 'WHERE filetime > ? order by filetime asc',
#                                 (tstop,))
#         if not query_row:
#             query_row = db.fetchone('SELECT tstop, rowstop FROM archfiles '
#                                     'order by filetime desc')

#         rowstop = query_row['rowstop']

#         return slice(rowstart, rowstop)

#     return get_interval_from_db(tstart, tstop, _split_path(msid_files['archfiles'].abs))


@contextlib.contextmanager
def _cache_ft():
    """
    Cache the global filetype ``ft`` context variable so that fetch operations
    do not corrupt user values of ``ft``.
    """
    ft_cache_pickle = pickle.dumps(ft)
    try:
        yield
    finally:
        ft_cache = pickle.loads(ft_cache_pickle)
        ft.update(ft_cache)
        delkeys = [x for x in ft if x not in ft_cache]
        for key in delkeys:
            del ft[key]


def _fix_ctu_dwell_mode_bads(msid, bads):
    """
    Because of an issue related to the placement of the dwell mode flag, MSIDs that get
    stepped on in dwell mode get a bad value at the beginning of a dwell mode, while the
    dwell mode values (DWELLnn) get a bad value at the end.  This does a simple
    brute-force fix of expanding any section of bad values by ones sample in the
    appropriate direction.
    """
    MSID = msid.upper()
    stepped_on_msids = ('4PRT5BT', '4RT585T', 'AFLCA3BI', 'AIRU1BT', 'CSITB5V',
                        'CUSOAOVN', 'ECNV3V', 'PLAED4ET', 'PR1TV01T', 'TCYLFMZM',
                        'TOXTSUPN', 'ES1P5CV', 'ES2P5CV')

    if MSID in stepped_on_msids or re.match(r'DWELL\d\d', MSID):
        # Find transitions from good value to bad value.  Turn that
        # good value to bad to extend the badness by one sample.
        ok = (bads[:-1] == False) & (bads[1:] == True)
        bads[:-1][ok] = True

    return bads


def add_logging_handler(level=logging.INFO,
                        formatter=None,
                        handler=None):
    """Configure logging for fetch module.

    :param level: logging level (logging.DEBUG, logging.INFO, etc)
    :param formatter: logging.Formatter (default: Formatter('%(funcName)s: %(message)s'))
    :param handler: logging.Handler (default: StreamHandler())
    """

    if formatter is None:
        formatter = logging.Formatter('%(funcName)s: %(message)s')

    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)


def _plural(x):
    """Return English plural of ``x``.  Super-simple and only valid for the
    known small set of cases within fetch where it will get applied.
    """
    return x + 'es' if (x.endswith('x') or x.endswith('s')) else x + 's'
