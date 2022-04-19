#!/usr/bin/env python
# see LICENSE.rst

from __future__ import print_function, division, absolute_import
from jeta.archive.operations import TELEMETRY_ARCHIVE

import re
import os
import glob
import time
import pickle
from random import seed
import shutil
import argparse
import itertools

from collections import OrderedDict, defaultdict, deque

from astropy.time import Time

import pyyaks.logger
import pyyaks.context

import h5py
import tables
import numpy as np
import scipy.stats.mstats

import jeta.archive.fetch as fetch
from jeta.archive.utils import get_env_variable


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Dry run (no actual file or database updates)")
    parser.add_argument("--no-full",
                        action="store_false",
                        dest="update_full",
                        default=True,
                        help=("Do not fetch files from archive and update "
                              "full-resolution MSID archive"))
    parser.add_argument("--no-stats",
                        action="store_false",
                        dest="update_stats",
                        default=True,
                        help="Do not update 5 minute and daily stats archive")
    parser.add_argument("--create",
                        action="store_true",
                        help="Create the MSID H5 files from scratch")
    parser.add_argument("--truncate",
                        help="Truncate archive after <date> (CAUTION!!)")
    parser.add_argument("--max-lookback-time",
                        type=float,
                        default=120,
                        help="Maximum look back time for updating statistics (days)")
    parser.add_argument("--date-now",
                        default=Time(Time.now()).yday,
                        help="Set effective processing date for testing (default=NOW)")
    parser.add_argument("--date-start",
                        default=None,
                        help=("Processing start date (loops by max-lookback-time "
                              "until date-now if set)"))
    parser.add_argument("--max-gap",
                        type=float,
                        help="Maximum time gap between archive files")
    parser.add_argument("--allow-gap-after-days",
                        type=float,
                        default=4,
                        help="Allow archive file gap when file is this old (days, default=4)")
    parser.add_argument("--max-arch-files",
                        type=int,
                        default=500,
                        help="Maximum number of archive files to ingest at once")
    parser.add_argument("--data-root",
                        default=".",
                        help="Engineering archive root directory for MSID and arch files")
    parser.add_argument("--occ",
                        action="store_true",
                        help="Running on the OCC GRETA network (no arc5gl)")
    parser.add_argument("--content",
                        action='append',
                        help="Content type to process [match regex] (default = all)")
    parser.add_argument("--log-level",
                        help="Logging level")
    parser.add_argument("--ingest-file-format",
                        default='h5',
                        choices={"h5", "csv"},
                        help=("Select the format of the ingest file type \
                             as either hdf5 or csv (default = h5)"))

    return parser.parse_args("")

#
ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')

#
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


JETA_LOGS = get_env_variable('JETA_LOGS')


# Configure fetch.MSID to cache recent results for performance in
# derived parameter updates.
fetch.CACHE = True


opt = get_options()
if opt.create:
    opt.update_stats = False

ft = fetch.ft


# Set up fetch so it will first try to read from opt.data_root if that is
# provided as an option and exists, and if not fall back to the default of
# fetch.ENG_ARCHIVE.  Fetch is a read-only process so this is safe when
# testing.
# if opt.data_root:
#     fetch.msid_files.basedir = ':'.join([opt.data_root, fetch.ENG_ARCHIVE])

# Set up logging
loglevel = pyyaks.logger.VERBOSE if opt.log_level is None else int(opt.log_level)

logger = pyyaks.logger.get_logger(
    filename=f'{JETA_LOGS}/jeta.update.log',
    name='jeta_logger',
    level=loglevel,
    format="%(asctime)s %(message)s"
)


# Also adjust fetch logging if non-default log-level supplied (mostly for debug)
if opt.log_level is not None:
    fetch.add_logging_handler(level=int(opt.log_level))


def main():
    """ Update stats for the data archive based on opt parameters.

    This may be called in a loop by the program-level main().
    """

    logger.info('Runtime options: \n{}'.format(opt))
    logger.info('Update Module: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch Module: {}'.format(os.path.abspath(fetch.__file__)))

    # TODO: Write a tests
    # colnames = [x for x in pickle.load('msids in the arhcive file', 'rb')) if x not in fetch.IGNORE_COLNAMES]

    colnames = None
    ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')
    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        colnames = list(h5.keys())

    if opt.update_stats:
        for colname in colnames:
            msid = statistics(colname, 'daily')
            statistics(colname, '5min', msid)
        logger.info(f'Stats updated.')


def del_stats(colname, time0, interval):
    """Delete all rows in ``interval`` stats file for column ``colname`` that
    occur after time ``time0`` - ``interval``.  This is used to fix problems
    that result from a file misorder.  Subsequent runs of update_stats will
    refresh the values correctly.
    """
    dt = {'5min': 328,
          'daily': 86400}[interval]

    ft['msid'] = colname
    ft['interval'] = interval
    stats_file = f"{TELEMETRY_ARCHIVE}/data/tlm/stats/{interval}/{str(colname).upper()}.h5"
    if not os.path.exists(stats_file):
        raise IOError('Stats file {} not found'.format(stats_file))

    logger.info('Fixing stats file %s after time %s', stats_file, Time(time0, format="unix").yday)

    stats = tables.open_file(stats_file, mode='a',
                            filters=tables.Filters(complevel=5, complib='zlib'))
    index0 = time0 // dt - 1
    indexes = stats.root.data.col('index')[:]
    row0 = np.searchsorted(indexes, [index0])[0] - 1
    if opt.dry_run:
        n_del = len(stats.root.data) - row0
    else:
        if row0 > 0:
            n_del = stats.root.data.remove_rows(row0, len(stats.root.data))
        else:
            n_del = len(stats.root.data)  
            stats.remove_node(stats.root.data)  #LITA-190
        
    logger.info('Deleted %d rows from row %s (%s) to end', n_del, row0,
                Time(indexes[row0] * dt, format='unix').yday)
    stats.close()


def calc_stats_vals(msid, rows, indexes, interval):
    """
    Compute statistics values for ``msid`` over specified intervals.
    :param msid: Msid object (filter_bad=True)
    :param rows: Msid row indices corresponding to stat boundaries
    :param indexes: Universal index values for stat (row times // dt)
    :param interval: interval name (5min or daily)
    """
    quantiles = (1, 5, 16, 50, 84, 95, 99)
    n_out = len(rows) - 1

    # Check if data type is "numeric".  Boolean values count as numeric,
    # partly for historical reasons, in that they support funcs like
    # mean (with implicit conversion to float).
    msid_dtype = msid.vals.dtype
    msid_is_numeric = issubclass(msid_dtype.type, (np.number, np.bool_))

    # Predeclare numpy arrays of correct type and sufficient size for accumulating results.
    out = OrderedDict()
    out['index'] = np.ndarray((n_out,), dtype=np.int32)
    out['n'] = np.ndarray((n_out,), dtype=np.int32)
    out['val'] = np.ndarray((n_out,), dtype=msid_dtype)

    if msid_is_numeric:
        out['min'] = np.ndarray((n_out,), dtype=msid_dtype)
        out['max'] = np.ndarray((n_out,), dtype=msid_dtype)
        out['mean'] = np.ndarray((n_out,), dtype=np.float32)

        if interval == 'daily':
            out['std'] = np.ndarray((n_out,), dtype=msid_dtype)
            for quantile in quantiles:
                out['p{:02d}'.format(quantile)] = np.ndarray((n_out,), dtype=msid_dtype)

    # MSID may have state codes
    # if msid.state_codes:
    #     for raw_count, state_code in msid.state_codes:
    #         out['n_' + fix_state_code(state_code)] = np.zeros(n_out, dtype=np.int32)

    i = 0
    for row0, row1, index in zip(rows[:-1], rows[1:], indexes[:-1]):
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
                    dts[1:-1] = ((times[1:-1] - times[:-2]) +
                                 (times[2:] - times[1:-1])) / 2.0
                    negs = dts < 0.0
                    if np.any(negs):
                        times_dts = [(Time(t, format="unix").yday, dt)
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

            # if msid.state_codes:
                # If MSID has state codes then count the number of values in each state
                # and store.  The MSID values can have trailing spaces to fill out to a
                # uniform length, so state_code is right padded accordingly.
                # max_len = max(len(state_code) for raw_count, state_code in msid.state_codes)
                # fmtstr = '{:' + str(max_len) + 's}'
                # for raw_count, state_code in msid.state_codes:
                #     state_count = np.count_nonzero(vals == fmtstr.format(state_code))
                #     out['n_' + fix_state_code(state_code)][i] = state_count

            i += 1

    return np.rec.fromarrays([x[:i] for x in out.values()], names=list(out.keys()))


def statistics(colname, interval, msid=None):

    dt = {
        '5min': 328,
        'daily': 86400
    }[interval]

    ft['msid'] = colname
    ft['interval'] = interval

    stats_file = f"{TELEMETRY_ARCHIVE}/data/tlm/stats/{interval}/{str(colname).upper()}.h5"

    logger.info('Updating stats file %s', stats_file)
    
    if not os.path.exists(f'{TELEMETRY_ARCHIVE}/data/tlm/stats/{interval}/'):
        logger.info('Making stats dir {}'.format(f'{TELEMETRY_ARCHIVE}/data/tlm/stats/{interval}/'))
        os.makedirs(f'{TELEMETRY_ARCHIVE}/data/tlm/stats/{interval}/')

    stats = tables.open_file(
        stats_file,
        mode='a',
        filters=tables.Filters(complevel=5, complib='zlib')
    )

    # INDEX0 is somewhat before any CXC archive data (which starts around 1999:205)
    INDEX0 = Time('1999:200:00:00:00', format='yday').unix // dt
    try:
        index0 = stats.root.data.cols.index[-1] + 1
    except tables.NoSuchNodeError:
        index0 = INDEX0

    # Get all new data. time0 is the fetch start time which nominally starts at
    # 500 sec before the last available record.  However some MSIDs may not
    # be sampled for years at a time so once the archive is built and kept
    # up to date then do not look back beyond a certain point.
    if msid is None:
        # fetch telemetry plus a little extra

        time0 = max(Time(Time(Time.now()).yday, format="yday").unix - opt.max_lookback_time * 86400,
                    index0 * dt - 500)
        time1 = Time(Time.now()).yday

        msid = fetch.MSID(colname, Time(time0, format="unix").yday, time1, filter_bad=False)

    if len(msid.times) > 0:
        if index0 == INDEX0:
            # Must be creating the file, so back up a bit from earliest MSID data
            index0 = msid.times[0] // dt - 2

        indexes = np.arange(index0, msid.times[-1] / dt, dtype=np.int32)
        times = indexes * dt

        if len(times) > 2:
            rows = np.searchsorted(msid.times, times)
            vals_stats = calc_stats_vals(msid, rows, indexes, interval)
            if len(vals_stats) > 0:
                # Don't change the following logic in order to add stats data
                # on the same pass as creating the table.  Tried it and
                # something got broken so that there was a single bad record
                # after the first bunch.
                if not opt.dry_run:
                    try:
                        stats.root.data.append(vals_stats)
                        logger.info('  Adding %d records', len(vals_stats))
                    except tables.NoSuchNodeError:
                        logger.info('  Creating table with %d records ...', len(vals_stats))
                        stats.create_table(stats.root, 'data', vals_stats,
                                          "{} sampling".format(interval), expectedrows=2e7)
                    stats.root.data.flush()
            else:
                logger.info('  No stat records within available fetched values')
        else:
            logger.info('  No full stat intervals within fetched values')
    else:
        logger.info('  No MSID data found within {} to {}'
                    .format(msid.datestart, msid.datestop))

    stats.close()

    return msid


if __name__ == "__main__":
    main()
