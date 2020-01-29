#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Fix bad values that are in the engineering archive.

Typically this is needed because of bad telemetry that is not marked
as such in the CXC archive.  This code will mark a single value or
range of values as bad (via the quality flag) and re-run the statistics
computations.  The database (HDF5 file) values are updated in place.

Usage::

  fix_bad_values.py [-h] [--msid MSID] [--start START] [--stop STOP]
                           [--run] [--data-root DATA_ROOT]

  Fix bad values in eng archive

  optional arguments:
    -h, --help            show this help message and exit
    --msid MSID           MSID name
    --start START         Start time of bad values
    --stop STOP           Stop time of bad values (default is start + 1 msec)
    --run                 Actually modify files (dry run is the default)
    --data-root DATA_ROOT
                          Engineering archive root directory for MSID and arch
                          files

Example::

  # First dry run
  % ./fix_bad_values.py --msid aorate3 --start=2013:146:16:12:44.600 \
                        --data-root=/proj/sot/ska/data/eng_archive

  # Now the real thing
  % ./fix_bad_values.py --msid aorate3 --start=2013:146:16:12:44.600 \
                        --data-root=/proj/sot/ska/data/eng_archive --run

  % emacs NOTES.fix_bad_values  # copy the above into NOTES for the record

"""

import argparse
import itertools
import contextlib

import numpy as np
import tables
import scipy.stats.mstats

import pyyaks.context
import pyyaks.logger
from Ska.engarchive import fetch
import Ska.engarchive.file_defs as file_defs
from Chandra.Time import DateTime

ft = fetch.ft
opt = None
msid_files = None
logger = None


@contextlib.contextmanager
def _set_msid_files_basedir(datestart):
    """
    If datestart is before 2000:001:00:00:00 then use the 1999 archive files.
    """
    try:
        cache_basedir = msid_files.basedir
        if datestart < fetch.DATE2000_LO:
            # Note: don't use os.path.join because ENG_ARCHIVE and basedir must
            # use linux '/' convention but this might be running on Windows.
            msid_files.basedir = msid_files.basedir + '/1999'
        yield
    finally:
        msid_files.basedir = cache_basedir


def get_opt():
    parser = argparse.ArgumentParser(description='Fix bad values in eng archive')
    parser.add_argument('--msid',
                        type=str,
                        help='MSID name')
    parser.add_argument('--start',
                        help='Start time of bad values')
    parser.add_argument('--stop',
                        help='Stop time of bad values')
    parser.add_argument('--value',
                        help='Update with <value> instead of setting as bad')
    parser.add_argument("--run",
                        action="store_true",
                        help="Actually modify files (dry run is the default)")
    parser.add_argument("--data-root",
                        default=".",
                        help="Engineering archive root directory for MSID and arch files")

    args = parser.parse_args()
    return args


def calc_stats_vals(msid, rows, indexes, interval):
    quantiles = (1, 5, 16, 50, 84, 95, 99)
    cols_stats = ('index', 'n', 'val')
    n_out = len(rows) - 1
    msid_dtype = msid.vals.dtype
    msid_is_numeric = not msid_dtype.name.startswith('string')
    # Predeclare numpy arrays of correct type and sufficient size for accumulating results.
    out = dict(index=np.ndarray((n_out,), dtype=np.int32),
               n=np.ndarray((n_out,), dtype=np.int32),
               val=np.ndarray((n_out,), dtype=msid_dtype),
               )
    if msid_is_numeric:
        cols_stats += ('min', 'max', 'mean')
        out.update(dict(min=np.ndarray((n_out,), dtype=msid_dtype),
                        max=np.ndarray((n_out,), dtype=msid_dtype),
                        mean=np.ndarray((n_out,), dtype=np.float32),))
        if interval == 'daily':
            cols_stats += ('std',) + tuple('p%02d' % x for x in quantiles)
            out['std'] = np.ndarray((n_out,), dtype=msid_dtype)
            out.update(('p%02d' % x, np.ndarray((n_out,), dtype=msid_dtype)) for x in quantiles)
    i = 0
    for row0, row1, index in itertools.izip(rows[:-1], rows[1:], indexes[:-1]):
        vals = msid.vals[row0:row1]
        times = msid.times[row0:row1]
        n_vals = len(vals)
        if n_vals > 0:
            out['index'][i] = index
            out['n'][i] = n_vals
            out['val'][i] = vals[n_vals // 2]
            if msid_is_numeric:
                if n_vals <= 2:
                    dts = np.ones(n_vals, dtype=np.float64)
                else:
                    dts = np.empty(n_vals, dtype=np.float64)
                    dts[0] = times[1] - times[0]
                    dts[-1] = times[-1] - times[-2]
                    dts[1:-1] = ((times[1:-1] - times[:-2])
                                 + (times[2:] - times[1:-1])) / 2.0
                    negs = dts < 0.0
                    if np.any(negs):
                        times_dts = [(DateTime(t).date, dt)
                                     for t, dt in zip(times[negs], dts[negs])]
                        logger.warning('WARNING - negative dts in {} at {}'
                                       .format(msid.MSID, times_dts))

                    # Clip to range 0.001 to 300.0.  The low bound is just there
                    # for data with identical time stamps.  This shouldn't happen
                    # but in practice might.  The 300.0 represents 5 minutes and
                    # is the largest normal time interval.  Data near large gaps
                    # will get a weight of 5 mins.
                    dts.clip(0.001, 300.0, out=dts)
                sum_dts = np.sum(dts)

                out['min'][i] = np.min(vals)
                out['max'][i] = np.max(vals)
                out['mean'][i] = np.sum(dts * vals) / sum_dts
                if interval == 'daily':
                    # biased weighted estimator of variance (N should be big enough)
                    # http://en.wikipedia.org/wiki/Mean_square_weighted_deviation
                    sigma_sq = np.sum(dts * (vals - out['mean'][i]) ** 2) / sum_dts
                    out['std'][i] = np.sqrt(sigma_sq)
                    quant_vals = scipy.stats.mstats.mquantiles(vals, np.array(quantiles) / 100.0)
                    for quant_val, quantile in zip(quant_vals, quantiles):
                        out['p%02d' % quantile][i] = quant_val
            i += 1

    return np.rec.fromarrays([out[x][:i] for x in cols_stats], names=cols_stats)


def fix_stats_h5(msid, tstart, tstop, interval):
    dt = {'5min': 328,
          'daily': 86400}[interval]

    ft['msid'] = msid
    ft['interval'] = interval
    datestart = DateTime(tstart).date
    with _set_msid_files_basedir(datestart):
        stats_file = msid_files['stats'].abs
    logger.info('Updating stats file {}'.format(stats_file))

    index0 = int(tstart // dt)
    index1 = int(tstop // dt) + 1
    indexes = np.arange(index0, index1 + 1, dtype=np.int32)
    times = indexes * dt
    logger.info('Indexes = {}:{}'.format(index0, index1))

    logger.info('Fetching {} data between {} to {}'.format(msid, DateTime(times[0] - 500).date,
                                                           DateTime(times[-1] + 500).date))
    dat = fetch.Msid(msid, times[0] - 500, times[-1] + 500)

    # Check within each stat interval?
    if len(dat.times) == 0:
        logger.info('Skipping: No values within interval {} to {}'
                    .format(DateTime(times[0] - 500).date,
                            DateTime(times[-1] + 500).date))
        return

    rows = np.searchsorted(dat.times, times)
    vals_stats = calc_stats_vals(dat, rows, indexes, interval)

    try:
        h5 = tables.openFile(stats_file, 'a')
        table = h5.root.data
        row0, row1 = np.searchsorted(table.col('index'), [index0, index1])
        for row_idx, vals_stat in itertools.izip(range(row0, row1), vals_stats):
            if row1 - row0 < 50 or row_idx == row0 or row_idx == row1 - 1:
                logger.info('Row index = {}'.format(row_idx))
                logger.info('  ORIGINAL: %s', table[row_idx])
                logger.info('  UPDATED : %s', vals_stat)
            if opt.run:
                table[row_idx] = tuple(vals_stat)
    finally:
        h5.close()

    logger.info('')


def fix_msid_h5(msid, tstart, tstop):
    """
    Fix the msid full-resolution HDF5 data file
    """
    logger.info('Fixing MSID {} h5 file'.format(msid))
    row_slice = fetch.get_interval(ft['content'].val, tstart, tstop)

    # Load the time values and find indexes corresponding to start / stop times
    ft['msid'] = 'TIME'

    filename = msid_files['data'].abs
    logger.info('Reading TIME file {}'.format(filename))

    h5 = tables.openFile(filename)
    times = h5.root.data[row_slice]
    h5.close()

    # Index values that need to be fixed are those within the specified time range, offset by
    # the beginning index of the row slice.
    fix_idxs = np.flatnonzero((tstart <= times) & (times <= tstop)) + row_slice.start

    # Open the msid HDF5 data file and set the corresponding quality flags to True (=> bad)
    ft['msid'] = msid
    filename = msid_files['msid'].abs
    logger.info('Reading msid file {}'.format(filename))

    h5 = tables.openFile(filename, 'a')
    try:
        if opt.value is not None:
            # Set data to <value> over the specified time range
            i0, i1 = fix_idxs[0], fix_idxs[-1] + 1
            logger.info('Changing {}.data[{}:{}] to {}'
                        .format(msid, i0, i1, opt.value))
            if opt.run:
                h5.root.data[i0:i1] = opt.value
        else:
            for idx in fix_idxs:
                quality = h5.root.quality[idx]
                if quality:
                    logger.info('Skipping idx={} because quality is already True'.format(idx))
                    continue
                if len(fix_idxs) < 100 or idx == fix_idxs[0] or idx == fix_idxs[-1]:
                    logger.info('{}.data[{}] = {}'.format(msid, idx, h5.root.data[idx]))
                    logger.info('Changing {}.quality[{}] from {} to True'
                                .format(msid, idx, quality))
                if opt.run:
                    h5.root.quality[idx] = True
    finally:
        h5.close()
    logger.info('')


def main():
    global opt
    global msid_files
    global logger

    opt = get_opt()

    # Set up infrastructure to directly access HDF5 files
    msid_files = pyyaks.context.ContextDict('msid_files',
                                            basedir=(opt.data_root or file_defs.msid_root))
    msid_files.update(file_defs.msid_files)

    # Set up fetch so it reads from opt.data_root
    fetch.msid_files.basedir = opt.data_root

    # Set up logging
    loglevel = pyyaks.logger.INFO
    logger = pyyaks.logger.get_logger(name='fix_bad_values', level=loglevel,
                                      format="%(message)s")

    logger.info('** If something gets corrupted then there is the NetApp snapshot for recovery **')
    logger.info('')
    if not opt.run:
        logger.info('** DRY RUN **')
        logger.info('')

    msid = opt.msid.upper()
    ft['content'] = fetch.content[msid]

    # Get the relevant row slice covering the requested time span for this content type
    tstart = DateTime(opt.start).secs - 0.001
    stop = DateTime(opt.stop or opt.start)
    tstop = stop.secs + 0.001

    # First fix the HDF5 file with full resolution MSID data
    # Need to potentially set basedir for 1999 data.
    with _set_msid_files_basedir(DateTime(tstart).date):
        fix_msid_h5(msid, tstart, tstop)

    # Now fix stats files
    fix_stats_h5(msid, tstart, tstop, '5min')
    fix_stats_h5(msid, tstart, tstop, 'daily')


if __name__ == '__main__':
    main()
