#!/usr/bin/env python
# see LICENSE.rst

from __future__ import print_function, division, absolute_import

import re
import os
import glob
import time
import pickle
import shutil
import argparse
import itertools

from collections import OrderedDict, defaultdict, deque

from astropy.time import Time
from Chandra.Time import DateTime

import Ska.File
import Ska.DBI
import Ska.Numpy
import pyyaks.logger
import pyyaks.context

import h5py
import tables
import numpy as np
import scipy.stats.mstats

import jeta.archive.fetch as fetch
import jeta.archive.file_defs as file_defs
import jeta.archive.derived as derived
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
                        default=60,
                        help="Maximum look back time for updating statistics (days)")
    parser.add_argument("--date-now",
                        default=DateTime().date,
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

    return parser.parse_args(args)


#
ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')

#
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')

#
INGEST_CADENCE = 2

# The expected life time of the mission in years
MISSION_LIFE_IN_YEARS = 20

# The maximum number of rows per file per analysis
MAX_ROWS_PER_FILE = 10_280_811


# An assumption about the average number of files per ingest
AVG_NUMBER_OF_FILES = 50

# Calculate the number of files in a given year for generation
# of data.
FILES_IN_A_YEAR = (AVG_NUMBER_OF_FILES * INGEST_CADENCE) * 365

# Configure fetch.MSID to cache recent results for performance in
# derived parameter updates.
fetch.CACHE = True


opt = get_options()
if opt.create:
    opt.update_stats = False

ft = fetch.ft

msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)
arch_files = pyyaks.context.ContextDict('update.arch_files',
                                        basedir=ENG_ARCHIVE)
arch_files.update(file_defs.arch_files)

# Set up fetch so it will first try to read from opt.data_root if that is
# provided as an option and exists, and if not fall back to the default of
# fetch.ENG_ARCHIVE.  Fetch is a read-only process so this is safe when
# testing.
if opt.data_root:
    fetch.msid_files.basedir = ':'.join([opt.data_root, fetch.ENG_ARCHIVE])

# Set up logging
loglevel = pyyaks.logger.VERBOSE if opt.log_level is None else int(opt.log_level)

logger = pyyaks.logger.get_logger(
    filename='/var/log/jeta.update.log',
    name='jeta_logger',
    level=loglevel,
    format="%(asctime)s %(message)s"
)


# Also adjust fetch logging if non-default log-level supplied (mostly for debug)
if opt.log_level is not None:
    fetch.add_logging_handler(level=int(opt.log_level))

_values = None
_times = None
_counts = None


def _create_msid_time_dataset(msid, h5):

    ft['msid'] = msid
    avg_msids_rows_per_ingest = len(_values[msid])

    expectedrows = (avg_msids_rows_per_ingest
                    * FILES_IN_A_YEAR
                    * MISSION_LIFE_IN_YEARS)
    h5shape = (0,)
    h5type = tables.Atom.from_dtype(np.dtype('float64'))
    filters = tables.Filters(complevel=5, complib='zlib')

    h5.create_earray(
        h5.root,
        'time',
        h5type,
        h5shape,
        title=msid,
        expectedrows=expectedrows,
        filters=filters
    )

    return 0


def _create_msid_value_dataset(msid, h5):

    ft['msid'] = msid
    avg_msids_rows_per_ingest = len(_values[msid])

    expectedrows = (avg_msids_rows_per_ingest
                    * FILES_IN_A_YEAR
                    * MISSION_LIFE_IN_YEARS)
    h5shape = (0,)

    # FIXME: The h5type fpr the msid values should vary
    # depending on the actual data to be stored.
    h5type = tables.Atom.from_dtype(np.dtype('float64'))
    filters = tables.Filters(complevel=5, complib='zlib')

    h5.create_earray(
        h5.root,
        'data',
        h5type,
        h5shape,
        title=msid,
        expectedrows=expectedrows,
        filters=filters
    )

    return 0


def _create_msid_directories(msids):
    """Create directories in the archive give a list of msids

        Parameters
        ----------
        msids : a python list of msids to add to the archive

    """

    for msid in msids:
        ft['msid'] = msid
        msid_directory_path = msid_files['msid'].abs
        if not os.path.exists(msid_directory_path):
            logger.info(f"Creating new archive directory for {msid} ...")
            os.makedirs(msid_directory_path)


def _create_archive_files(msids):
    for msid in msids:
        ft['msid'] = msid

        if not os.path.exists(msid_files['mnemonic_value'].abs):
            logger.info((
                f"Creating new archive files"
                f"(times.h5, values.h5) for {msid} ... "
            ))

            try:
                values_h5 = tables.open_file(
                    msid_files['mnemonic_value'].abs,
                    mode='w'
                )
                times_h5 = tables.open_file(
                    msid_files['mnemonic_times'].abs,
                    mode='w'
                )
                times_h5.close()
                values_h5.close()
            except Exception as err:
                print(err)


def _aggregate_dataset_samples(samples, large_sample, offset):
    """Aggregates the dataset samples into a single ndarry from an HDF5 group

        Parameters
        ----------
        samples : <class 'h5py._hl.group.Group'>
                  `samples` contains a collection of 1 - n
                   datasets of msid samples
        large_sample : ndarray - dtype=[
                                            ('id', '>i8'),
                                            ('observatoryTime', '>i8'),
                                            ('groundTime', '>i8'),
                                            ('apid', '>i2'),
                                            ('engineeringNumericValue', '>f8'),
                                            ('engineeringTextValue', 'S80'),
                                            ('alarmStatus', '>i2')
                                        ]
                        `large_sample` is a datastructure used to collect all
                        tlm data from different files in memory.
        offset : int
                 This parameter is used to track where in `large_sample`
                 to start appending new data.

        Returns
        -------
        large_sample
            A pointer to the data being collected
        offset
            A updated `offset` values base on the number of rows
            appended thus far in the ingest.
    """

    grouped_sample = np.empty((0,), dtype=[
        ('id', '>i8'),
        ('observatoryTime', '>i8'),
        ('groundTime', '>i8'),
        ('apid', '>i2'),
        ('engineeringNumericValue', '>f8'),
        ('engineeringTextValue', 'S80'),
        ('alarmStatus', '>i2')]
    )

    for i in range(1, len(samples)+1):
        data = samples[f'data{i}']
        dset = data[...]
        grouped_sample = np.concatenate((grouped_sample, dset))

    for i, v in enumerate(grouped_sample):
        large_sample[i + offset] = v

    offset += len(grouped_sample)

    return large_sample, offset


def _update_index_file(msid, epoch, index):
    """ Update the index file

        Parameters
        ----------
        filepath : str
            the index file path
        epoch : float
            the time
    """
    # filters = tables.Filters(complevel=5, complib='zlib')
    ft['msid'] = msid
    h5 = tables.open_file(
        msid_files['mnemonic_index'].abs,
        driver="H5FD_CORE",
        mode="a"
    )

    if index is not None and epoch is not None:

        try:
            if h5.__contains__('/epoch') is False:
                compound_datatype = np.dtype([
                    ('epoch', np.float64),
                    ('index', np.uint64),
                ])
                table = h5.create_table(h5.root, 'epoch', compound_datatype)
            else:
                table = h5.root.epoch
            table.row['index'] = index
            table.row['epoch'] = epoch
            table.row.append()
            table.flush()
        except Exception as err:
            raise ValueError(f"Could not create epoch: {err}")
        finally:
            h5.close()


def get_colnames():
    """Get column names for the current content type (defined by ft['content'])
    """

    colnames = [x for x in pickle.load(open(msid_files['colnames'].abs, 'rb'))
                if x not in fetch.IGNORE_COLNAMES]
    return colnames


def create_content_dir():
    """ Make empty files for colnames, pkl, and archive.meta.info.db3

    Must supply the --create option.
    """

    dirname = msid_files['contentdir'].abs

    if not os.path.exists(dirname):
        logger.info('Making directory {}'.format(dirname))
        os.makedirs(dirname)

    empty = set()
    if not os.path.exists(msid_files['colnames'].abs):
        with open(msid_files['colnames'].abs, 'wb') as f:
            pickle.dump(empty, f, protocol=0)

    if not os.path.exists(msid_files['colnames_all'].abs):
        with open(msid_files['colnames_all'].abs, 'wb') as f:
            pickle.dump(empty, f, protocol=0)

    if not os.path.exists(msid_files['archfiles'].abs):
        archfiles_def = open(
                        get_env_variable('ARCHIVE_DEFINITION_SOURCE')
            ).read()
        filename = msid_files['archfiles'].abs
        logger.info('Creating db {}'.format(filename))
        db = Ska.DBI.DBI(dbi='sqlite', server=filename, autocommit=False)
        db.execute(archfiles_def)
        db.commit()

    if not os.path.exists(msid_files['processed_files_directory'].abs):
        os.makedirs(msid_files['processed_files_directory'].abs)


_fix_state_code_cache = {}


def fix_state_code(state_code):
    """
    Return a version of ``state_code`` that has only alphanumeric chars.  This
    can be used as a column name, unlike e.g. "n_+1/2".  Since this gets called
    in an inner loop cache the result.
    """
    try:
        out = _fix_state_code_cache[state_code]
    except KeyError:
        out = state_code
        for sub_in, sub_out in ((r'\+', 'PLUS_'),
                                (r'\-', 'MINUS_'),
                                (r'>', '_GREATER_'),
                                (r'/', '_DIV_')):
            out = re.sub(sub_in, sub_out, out)
        _fix_state_code_cache[state_code] = out

    return out


def main():
    """ Perform one full update of the data archive based on opt parameters.

    This may be called in a loop by the program-level main().
    """

    logger.info('Runtime options: \n{}'.format(opt))
    logger.info('Update Module: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch Module: {}'.format(os.path.abspath(fetch.__file__)))
    logger.info('Syncing telemetry archive ...')

    # Get the archive content filetypes
    # Ex. filetypes = [('TELEM', 'TLM')]
    filetypes = fetch.filetypes

    if opt.content:
        contents = [x.upper() for x in opt.content]
        filetypes = [x for x in filetypes
                     if any(re.match(y, x.content) for y in contents)]

    # Update archive currently cannot create derived parameter content types
    if opt.create:
        filetypes = [x for x in filetypes if not x.content.startswith('DP_')]

    for filetype in filetypes:
        # Update attributes of global ContextValue "ft".  This is needed for
        # rendering of "files" ContextValue.
        ft['content'] = filetype.content.lower()

        if opt.create:
            create_content_dir()

        if not os.path.exists(msid_files['colnames'].abs):
            logger.info(f'No colnames.pickle for {ft["content"]} - skipping')
            continue

        if not os.path.exists(msid_files['archfiles'].abs):
            logger.info(f'No {msid_files["archfiles"].abs} for %s - skipping' % ft['content'])
            continue

        colnames = [x for x in pickle.load(open(msid_files['colnames'].abs, 'rb'))
                    if x not in fetch.IGNORE_COLNAMES]

        logger.info('Processing %s content type', ft['content'])

        if opt.truncate:
            truncate_archive(filetype, opt.truncate)
            continue

        if opt.update_full:
            if filetype['instrum'] == 'DERIVED':
                update_derived(filetype)
            else:
                update_archive(filetype)

        if opt.update_stats:
            for colname in colnames:
                msid = update_stats(colname, 'daily')
                update_stats(colname, '5min', msid)

        logger.info(f'SUCCESS: Telemetry Archive Sync Complete.')


def fix_misorders(filetype):
    """Fix problems in the eng archive where archive files were ingested out of
    time order.  This results in a non-monotonic times in the MSID hdf5 files
    and subsequently corrupts the stats files.  This routine looks for
    discontinuities in rowstart assuming filename ordering and swaps neighbors.
    One needs to verify in advance (--dry-run --fix-misorders --content ...)
    that this will be an adequate fix.
    Example::
      update_archive.py --dry-run --fix-misorders --content misc3eng
      update_archive.py --fix-misorders --content misc3eng >& fix_misc3.log
      update_archive.py --content misc3eng --max-lookback-time 100 >>& fix_misc3.log
    In the --dry-run it is important to verify that the gap is really just from
    two mis-ordered files that can be swapped.  Look at the rowstart,rowstop values
    in the filename-ordered list.
    :param filetype: filetype
    :returns: minimum time for all misorders found
    """
    colnames = pickle.load(open(msid_files['colnames'].abs, 'rb'))

    # Setup db handle with autocommit=False so that error along the way aborts insert transactions
    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs, autocommit=False)

    # Get misordered archive files
    archfiles = db.fetchall('SELECT * FROM archfiles order by filename')
    bads = archfiles['rowstart'][1:] - archfiles['rowstart'][:-1] < 0

    if not np.any(bads):
        logger.info('No misorders')
        return

    for bad in np.flatnonzero(bads):
        i2_0, i1_0 = archfiles['rowstart'][bad:bad + 2]
        i2_1, i1_1 = archfiles['rowstop'][bad:bad + 2]

        # Update hdf5 file for each column (MSIDs + TIME, MJF, etc)
        for colname in colnames:
            ft['msid'] = colname
            logger.info('Fixing %s', msid_files['msid'].abs)
            if not opt.dry_run:
                filepath = msid_files['mnemonic_value'].abs
                h5 = tables.open_file(filepath, mode='a')
                #h5 = tables.open_file(msid_files['msid'].abs, mode='a')
                hrd = h5.root.data
                hrq = h5.root.quality

                hrd1 = hrd[i1_0:i1_1]
                hrd2 = hrd[i2_0:i2_1]
                hrd[i1_0:i1_0 + len(hrd2)] = hrd2
                hrd[i1_0 + len(hrd2): i2_1] = hrd1

                hrq1 = hrq[i1_0:i1_1]
                hrq2 = hrq[i2_0:i2_1]
                hrq[i1_0:i1_0 + len(hrq2)] = hrq2
                hrq[i1_0 + len(hrq2): i2_1] = hrq1

                h5.close()

        # Update the archfiles table
        cmd = 'UPDATE archfiles SET '
        cols = ['rowstart', 'rowstop']
        cmd += ', '.join(['%s=?' % x for x in cols])
        cmd += ' WHERE filename=?'
        rowstart1 = i1_0
        rowstop1 = rowstart1 + i2_1 - i2_0
        rowstart2 = rowstop1 + 1
        rowstop2 = i2_1
        vals1 = [rowstart1, rowstop1, archfiles['filename'][bad]]
        vals2 = [rowstart2, rowstop2, archfiles['filename'][bad + 1]]
        logger.info('Running %s %s', cmd, vals1)
        logger.info('Running %s %s', cmd, vals2)

        logger.info('Swapping rows %s for %s', [i1_0, i1_1, i2_0, i2_1], filetype.content)
        logger.info('%s', archfiles[bad - 3:bad + 5])
        logger.info('')

        if not opt.dry_run:
            db.execute(cmd, [x.tolist() for x in vals1])
            db.execute(cmd, [x.tolist() for x in vals2])
            db.commit()

    return np.min(archfiles['tstart'][bads])


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
    stats_file = msid_files['stats'].abs
    if not os.path.exists(stats_file):
        raise IOError('Stats file {} not found'.format(stats_file))

    logger.info('Fixing stats file %s after time %s', stats_file, DateTime(time0).date)

    stats = tables.open_file(stats_file, mode='a',
                            filters=tables.Filters(complevel=5, complib='zlib'))
    index0 = time0 // dt - 1
    indexes = stats.root.data.col('index')[:]
    row0 = np.searchsorted(indexes, [index0])[0] - 1
    if opt.dry_run:
        n_del = len(stats.root.data) - row0
    else:
        n_del = stats.root.data.removeRows(row0, len(stats.root.data))
    logger.info('Deleted %d rows from row %s (%s) to end', n_del, row0,
                DateTime(indexes[row0] * dt).date)
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
    if msid.state_codes:
        for raw_count, state_code in msid.state_codes:
            out['n_' + fix_state_code(state_code)] = np.zeros(n_out, dtype=np.int32)

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

            if msid.state_codes:
                # If MSID has state codes then count the number of values in each state
                # and store.  The MSID values can have trailing spaces to fill out to a
                # uniform length, so state_code is right padded accordingly.
                max_len = max(len(state_code) for raw_count, state_code in msid.state_codes)
                fmtstr = '{:' + str(max_len) + 's}'
                for raw_count, state_code in msid.state_codes:
                    state_count = np.count_nonzero(vals == fmtstr.format(state_code))
                    out['n_' + fix_state_code(state_code)][i] = state_count

            i += 1

    return np.rec.fromarrays([x[:i] for x in out.values()], names=list(out.keys()))


def update_stats(colname, interval, msid=None):

    dt = {'5min': 328,
          'daily': 86400}[interval]

    ft['msid'] = colname
    ft['interval'] = interval
    stats_file = msid_files['stats'].abs

    logger.info('Updating stats file %s', stats_file)

    if not os.path.exists(msid_files['statsdir'].abs):
        logger.info('Making stats dir {}'.format(msid_files['statsdir'].abs))
        os.makedirs(msid_files['statsdir'].abs)

    stats = tables.open_file(stats_file, mode='a',
                            filters=tables.Filters(complevel=5, complib='zlib'))

    # INDEX0 is somewhat before any CXC archive data (which starts around 1999:205)
    INDEX0 = DateTime('1999:200:00:00:00').secs // dt
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

        time0 = max(DateTime(opt.date_now).secs - opt.max_lookback_time * 86400,
                    index0 * dt - 500)
        time1 = DateTime(opt.date_now).secs

        msid = fetch.MSID(colname, time0, time1, filter_bad=False)

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


def update_derived(filetype):
    """Update full resolution MSID archive files for derived parameters with ``filetype``
    """
    # Get the last H5 table row from archfiles table for this content type
    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs)
    last_row = db.fetchone('SELECT * FROM archfiles ORDER BY filetime DESC')

    # Set the starting index from the last row in archfiles.  This
    # uses Python slicing conventions so that the previous "end"
    # value is exactly the next "start" values, e.g. [index0:index1]
    # For derived parameters we have stopmjf <==> index1
    index0 = last_row['stopmjf']

    # Get the full set of rootparams for all colnames
    colnames = pickle.load(open(msid_files['colnames'].abs, 'rb'))
    colnames = [x for x in colnames if x.startswith('DP_')]
    msids = set()
    for colname in colnames:
        dp_class = getattr(derived, colname)
        dp = dp_class()
        msids = msids.union([x.upper() for x in dp.rootparams])
        time_step = dp.time_step  # will be the same for every DP

    # Find the last time in archive for each of the content types
    # occuring in the list of rootparam MSIDs.
    # fetch.content is a mapping from MSID to content type
    last_times = {}
    ft_content = ft['content'].val
    for msid in msids:
        ft['msid'] = 'TIME'
        content = ft['content'] = fetch.content[msid]
        if content not in last_times:
            h5 = tables.open_file(fetch.msid_files['msid'].abs, mode='r')
            last_times[content] = h5.root.data[-1]
            h5.close()
    last_time = min(last_times.values()) - 1000
    ft['content'] = ft_content

    # Make a list of indexes that will correspond to the index/time ranges
    # for each pseudo-"archfile".  In this context an archfile just specifies
    # the time range covered by an ingest, but is needed by fetch to roughly
    # locate rows in the H5 file for fast queries.  Each archfile is 10000 sec
    # long, and when updating the database no more than 1000000 seconds of
    # telemetry will be read at one time.
    archfile_time_step = 10000.0
    max_archfiles = int(1000000.0 / archfile_time_step)

    # Read data out to either date_now or the last available time in telemetry.
    # opt.date_now could be set in the past for testing.
    index_step = int(round(archfile_time_step / time_step))
    time1 = min(DateTime(opt.date_now).secs, last_time)
    index1 = int(time1 / time_step)
    indexes = np.arange(index0, index1, index_step)

    archfiles = []
    for index0, index1 in zip(indexes[:-1], indexes[1:]):
        archfiles.append('{}:{}:{}'.format(filetype['content'], index0, index1))
        if len(archfiles) == max_archfiles or index1 == indexes[-1]:
            update_telemetry_archive(archfiles)
            logger.verbose('update_telemetry_archive(archfiles={})'
                           .format(archfiles))
            archfiles = []


def update_archive(filetype):
    """ Get new ingest files for `filetype` and update telemetry archive
        from staged file data.

        Parameter
        =========

        filetype: the source (i.e. prod, test) type of file to be ingested to a
        `filetype` directory.
    """

    ingest_files = _sort_ingest_files_by_start_time(get_archive_files())

    if ingest_files:
        tstart = ingest_files[0]['tstart']
        tstop = ingest_files[-1]['tstop']

        logger.info(
            (
                f"Data Time Coverage (tstart, tstop): "
                f"({Time(tstart, format='unix').iso},"
                f"{Time(tstop, format='unix').iso})"
            )
        )

    # files_to_process = [
    #     ingest_file['filename']
    #     for ingest_file in ingest_files
    # ]

    if ingest_files:
        processed_ingest_files = process_ingest_files(
            ingest_files,
            tstart,
            tstop,
            chunk=6
        )

        # processed_ingest_files = update_telemetry_archive(files_to_ingest)
        # move_archive_files(filetype, processed_ingest_files)
    else:
        logger.info('No ingest files discovered in {}')


def make_h5_col_file_derived(dats, colname):
    """Make a new h5 table to hold column from ``dat``."""
    filename = msid_files['msid'].abs
    filedir = os.path.dirname(filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    # Estimate the number of rows for 20 years based on available data
    times = np.hstack([x['TIME'] for x in dats])
    dt = np.median(times[1:] - times[:-1])
    n_rows = int(86400 * 365 * 20 / dt)

    filters = tables.Filters(complevel=5, complib='zlib')
    h5 = tables.open_file(filename, mode='w', filters=filters)

    col = dats[-1][colname]
    h5shape = (0,) + col.shape[1:]
    h5type = tables.Atom.from_dtype(col.dtype)
    h5.create_earray(h5.root, 'data', h5type, h5shape, title=colname,
                    expectedrows=n_rows)
    h5.create_earray(h5.root, 'quality', tables.BoolAtom(), (0,), title='Quality',
                    expectedrows=n_rows)
    logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
                   .format(filename, colname, h5shape, n_rows / 1.0e6))
    h5.close()


def append_h5_col_derived(dats, colname):
    """Append new values to an HDF5 MSID data table.
    :param dats: List of pyfits HDU data objects
    :param colname: column name
    """
    def i_colname(dat):
        """Return the index for `colname` in `dat`"""
        return list(dat.dtype.names).index(colname)

    h5 = tables.open_file(msid_files['msid'].abs, mode='a')
    stacked_data = np.hstack([x[colname] for x in dats])
    stacked_quality = np.hstack([x['QUALITY'][:, i_colname(x)] for x in dats])
    logger.verbose('Appending %d items to %s' % (len(stacked_data), msid_files['msid'].abs))

    if not opt.dry_run:
        h5.root.data.append(stacked_data)
        h5.root.quality.append(stacked_quality)

    data_len = len(h5.root.data)
    h5.close()

    return data_len


def init_mnemonic_index_file(idx=None, epoch=None):

    compound_datatype = np.dtype([
        ('epoch', np.float64),
        ('index', np.uint64),
    ])

    with h5py.File(msid_files['mnemonic_index'].abs, mode='a', driver="core", backing_store=True) as h5:
        dset = h5.create_dataset('epoch', shape=(1,), compression="gzip", dtype=compound_datatype, chunks=True, maxshape=(None,))
        mnenmonic_idx = np.array([(epoch, idx)], dtype=[('epoch', 'f8'), ('index', np.uint64)])
        dset[:] = mnenmonic_idx
        h5.close()


def make_h5_col_file_tlm(dats, colname, metadata):
    """Make a new h5 table to hold column from ``dat``."""
    values_filename = msid_files['mnemonic_value'].abs
    # times_filename = msid_files['mnemonic_times'].abs

    filedir = os.path.dirname(values_filename)

    if not os.path.exists(filedir):
        os.makedirs(filedir)

    # Estimate the number of rows for 20 years based on available data
    # times = np.hstack([x['observatoryTime']/1000 for x in dats])

    # print(times[0])
    # print(Time(times[0], format='unix').iso)

    # dt = 1000  # np.median(times[1:] - times[:-1])

    n_rows = int(2500 * 24 * 365 * 20)
    filters = tables.Filters(complevel=5, complib='zlib')
    values_h5 = tables.open_file(values_filename, mode='w', filters=filters)
    # times_h5 = tables.open_file(times_filename, mode='w', filters=filters)

    # col = dats[-1][metadata[colname]]
    # # init_mnemonic_index_file(dat[colname]['index']['index'], dat[colname]['index']['epoch'])

    h5shape = (0,)
    h5type = tables.Atom.from_dtype(np.dtype('float64'))

    values_h5.create_earray(
        values_h5.root,
        'data',
        h5type,
        h5shape,
        title=colname,
        expectedrows=n_rows
    )
    # h5.create_earray(h5.root, 'quality', tables.BoolAtom(), (0,), title='Quality',
    #                  expectedrows=n_rows)
    logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
                   .format(values_filename, colname, h5shape, n_rows / 1.0e6))
    values_h5.close()


def _append_h5_col_tlm(msids):

    """Append new values to an HDF5 MSID data table.

    Parameters
    ----------
    msids : <class 'list'> of msids with data buffered for appending to the
            archive.
    """

    global _times
    global _values

    for msid in msids:

        ft['msid'] = msid

        values_h5 = tables.open_file(
            msid_files['mnemonic_value'].abs,
            mode='a'
        )

        if values_h5.__contains__('/data') is False:
            _create_msid_value_dataset(msid, values_h5)

        times_h5 = tables.open_file(
            msid_files['mnemonic_times'].abs,
            mode='a'
        )

        if times_h5.__contains__('/time') is False:
            _create_msid_time_dataset(msid, times_h5)

        # Check that 0 data is being appended.
        # if len(_times[msid]) == 0:
        #     _times.pop(msid, None)
        #     _values.pop(msid, None)
        #     values_h5.close()
        #     times_h5.close()
        #     continue

        # Check that time/value pairs match in length
        # if len(_times[msid]) != len(_values[msid]):
        #     values_h5.close()
        #     times_h5.close()
        #     print(_counts[msid])
        #     raise ValueError(f"ERROR: FULL STOP BECAUSE DATA IS CORRUPT, MISMATCH FOR {msid} !!!! ")

        # TODO: Verify epoch is correct
        epoch = Time(_times[msid][0]/1000.0, format='unix').jd

        # Index should point to current number of rows
        index = values_h5.root.data.nrows

        _times[msid] = get_delta_times(_times[msid], epoch)

        if not opt.dry_run:
            values_h5.root.data.append(_values[msid])
            times_h5.root.time.append(_times[msid])

        _update_index_file(msid, epoch, index)

        values_h5.close()
        times_h5.close()

    return 0


def truncate_archive(filetype, date):
    """Truncate msid and statfiles for every archive file after date (to nearest
    year:doy)
    """
    colnames = pickle.load(open(msid_files['colnames'].abs, 'rb'))

    date = DateTime(date).date
    year, doy = date[0:4], date[5:8]

    # Setup db handle with autocommit=False so that error along the way aborts insert transactions
    db = Ska.DBI.DBI(
        dbi='sqlite',
        server=msid_files['archfiles'].abs,
        autocommit=False
    )

    # Get the earliest row number from the archfiles table where year>=year and doy=>doy
    out = db.fetchall('SELECT rowstart FROM archfiles '
                      'WHERE year>={0} AND doy>={1}'.format(year, doy))
    if len(out) == 0:
        return
    rowstart = out['rowstart'].min()
    time0 = DateTime("{0}:{1}:00:00:00".format(year, doy)).secs

    for colname in colnames:
        ft['msid'] = colname
        filename = msid_files['mnemonic_value'].abs # msid_files['msid'].abs
        if not os.path.exists(filename):
            raise IOError('MSID file {} not found'.format(filename))
        if not opt.dry_run:
            h5 = tables.open_file(filename, mode='a')
            h5.root.data.truncate(rowstart)
            h5.root.quality.truncate(rowstart)
            h5.close()
        logger.verbose('Removed rows from {0} for filetype {1}:{2}'.format(
            rowstart, filetype['content'], colname))

        # Delete the 5min and daily stats, with a little extra margin
        if colname not in fetch.IGNORE_COLNAMES:
            del_stats(colname, time0, '5min')
            del_stats(colname, time0, 'daily')

    cmd = 'DELETE FROM archfiles WHERE (year>={0} AND doy>={1}) OR year>{0}'.format(year, doy, year)
    if not opt.dry_run:
        db.execute(cmd)
        db.commit()
    logger.verbose(cmd)


def is_file_already_in_db(ingest_file_path, db):

    filename = os.path.basename(ingest_file_path)
    if db.fetchall('SELECT filename FROM archfiles WHERE filename=?', (filename,)):
        logger.verbose('File %s already in archfiles - unlinking and skipping' % filename)
        os.unlink(ingest_file_path)
        return True


def read_archfile(i, f, filetype, row, colnames, archfiles, db):
    """Read filename ``f`` with index ``i`` (position within list of filenames).  The
    file has type ``filetype`` and will be added to MSID file at row index ``row``.
    ``colnames`` is the list of column names for the content type (not used here).
    """

    # Check if filename is already in archfiles.  If so then abort further processing.
    filename = os.path.basename(f)
    # dat = defaultdict(list)

    if db.fetchall('SELECT filename FROM archfiles WHERE filename=?', (filename,)):
        logger.verbose('File %s already in archfiles - unlinking and skipping' % f)
        os.unlink(f)
        return None, None

    # Read HDF5 or CVS ingest file and accumulate data into dats list and header into headers dict
    logger.info('Reading (%d / %d) %s' % (i, len(archfiles), filename))

    import h5py

    h5 = h5py.File(f, 'r')
    all_samples, tstart, tstop = concat_fos_samples(h5['samples'])
    metadata = h5['metadata'][...]

    # Accumlate relevant info about archfile that will be ingested into
    # MSID h5 files.  Commit info before h5 ingest so if there is a failure
    # the needed info will be available to do the repair.
    archfiles_row = dict(filename=f,
                         tstart=start_time,
                         tstop=stop_time,
                         rowstart=row,
                         rowstop=row + 1,
                         year=f[f.find('-') + 1:f.find('-') + 5],
                         doy=f[f.rfind('-') + 1:f.rfind('-') + 4],
                         date=Time.now().iso)

    h5.close()

    return all_samples, archfiles_row, metadata


def read_derived(i, filename, filetype, row, colnames, archfiles, db):
    """Read derived data using eng_archive and derived computation classes.
    ``filename`` has format <content>_<index0>_<index1> where <content>
    is the content type (e.g. "dp_thermal128"), <index0> is the start index for
    the new data and index1 is the end index (using Python slicing convention
    index0:index1).  Args ``i``, ``filetype``, and ``row`` are as in
    read_archive().  ``row`` must equal <index0>.  ``colnames`` is the list of
    column names for the content type.
    """
    # Check if filename is already in archfiles.  If so then abort further processing.

    if db.fetchall('SELECT filename FROM archfiles WHERE filename=?', (filename,)):
        logger.verbose('File %s already in archfiles - skipping' % filename)
        return None, None

    # f has format <content>_<index0>_<index1>
    # <content> has format dp_<content><mnf_step> e.g. dp_thermal128
    content, index0, index1 = filename.split(':')
    index0 = int(index0)
    index1 = int(index1)
    mnf_step = int(re.search(r'(\d+)$', content).group(1))
    time_step = mnf_step * derived.MNF_TIME
    times = time_step * np.arange(index0, index1)

    logger.info('Reading (%d / %d) %s' % (i, len(archfiles), filename))
    vals = {}
    bads = np.zeros((len(times), len(colnames)), dtype=np.bool)
    for i, colname in enumerate(colnames):
        if colname == 'TIME':
            vals[colname] = times
            bads[:, i] = False
        else:
            dp_class = getattr(Ska.engarchive.derived, colname.upper())
            dp = dp_class()
            dataset = dp.fetch(times[0] - 1000, times[-1] + 1000)
            ok = (index0 <= dataset.indexes) & (dataset.indexes < index1)
            vals[colname] = dp.calc(dataset)[ok]
            bads[:, i] = dataset.bads[ok]

    vals['QUALITY'] = bads
    dat = Ska.Numpy.structured_array(vals, list(colnames) + ['QUALITY'])

    # Accumlate relevant info about archfile that will be ingested into
    # MSID h5 files.  Commit info before h5 ingest so if there is a failure
    # the needed info will be available to do the repair.
    date = DateTime(times[0]).date
    year, doy = date[0:4], date[5:8]
    archfiles_row = dict(filename=filename,
                         filetime=int(index0 * time_step),
                         year=year,
                         doy=doy,
                         tstart=times[0],
                         tstop=times[-1],
                         rowstart=row,
                         rowstop=row + len(dat),
                         startmjf=index0,
                         stopmjf=index1,
                         date=date)

    return dat, archfiles_row


def get_dat_colnames(dat):
    """Iteratable over dat colnames"""
    return dat if isinstance(dat, dict) else dat.dtype.names


def get_delta_times(times, epoch=None):

    if epoch is None:
        raise ValueError("Must have epoch")
    # epoch = Time(time.gmtime(0), format='unix').jd

    x = [t/1000.0 for t in times]
    jd_times = Time(x, format='unix').jd
    return np.diff(np.insert(jd_times, 0, epoch))


def _allocate_large_sample(preallocation_size):

    return np.empty((preallocation_size,), dtype=[
        ('id', '>i8'),
        ('observatoryTime', '>i8'),
        ('groundTime', '>i8'),
        ('apid', '>i2'),
        ('engineeringNumericValue', '>f8'),
        ('engineeringTextValue', 'S80'),
        ('alarmStatus', '>i2')
    ])


def _organize_data_for_append(large_sample, mdmap):

    global _times
    global _values
    global _counts

    missing_ids = []

    for idx, data in enumerate(large_sample):
        if data['id'] == 0:
            continue
        try:
            name = mdmap[data['id']]
            _values[name].append(data['engineeringNumericValue'])
            # times are in milliseonds since unix epoch
            _times[name].append(data['observatoryTime'])
            # _counts[name] = _counts[name] + len(_times[name])
        except Exception as err:
            logger.error(f"Error: {err}")
            missing_ids.append(data['id'])

    return missing_ids


def _sort_ingest_files_by_start_time(list_of_files=[]):
    ingest_list = []

    for file in list_of_files:
        with h5py.File(file, 'r') as f:

            tstart = f['samples']["data1"].attrs['dataStartTime'][0]/1000
            tstop = f['samples'][f"data{len(f['samples'])}"].attrs['dataStopTime'][0]/1000
            ingest_list.append(
                {
                    'filename': f.filename,
                    'tstart': tstart,
                    'tstop': tstop,
                    'numPoints': f.attrs['/numPoints']
                }
            )

    return sorted(ingest_list, key=lambda k: k['tstart'])


def reset_storage():
    global _values
    _values = defaultdict(list)

    global _times
    _times = defaultdict(list)

    global _counts
    _counts = defaultdict(int)


def process_ingest_files(files_to_process, tstart, tstop, chunk=4):

    processing_start_time = Time(Time.now(), format="datetime").iso

    file_processing_queue = deque(files_to_process)
    original_queue_length = len(file_processing_queue)

    processed_files = []

    while len(file_processing_queue) != 0:

        mdmap = {}
        msids = []
        offset = 0

        reset_storage()

        if len(file_processing_queue) < chunk:
            chunk = len(file_processing_queue)

        file_processing_chunk = [
            file_processing_queue.popleft()
            for i in range(chunk)
        ]

        # Sum the number of points for a chunk to get pre-allocation value
        num_points_in_chunk = sum([i['numPoints'] for i in file_processing_chunk])
        num_points_in_chunk = num_points_in_chunk + (num_points_in_chunk * .01)

        large_sample = _allocate_large_sample(
            int(num_points_in_chunk)
        )

        metadata = np.empty((0,), dtype=[
            ('name', 'S80'),
            ('id', '>i8'),
            ('hasEngNumeric', '>i2'),
            ('hasEngText', '>i2')
        ])

        logger.info(
            f"Processing chunk of {len(file_processing_chunk)} ingest files.\n"
            f"{len(processed_files)} of {original_queue_length} files have been processed."
        )

        for ingest_file in file_processing_chunk:

            f = h5py.File(ingest_file['filename'], 'r')
            large_sample, offset = _aggregate_dataset_samples(f['samples'], large_sample, offset)
            metadata = np.unique(np.concatenate((metadata, f['metadata'][...]), 0))

            f.close()

        mdmap = {id:name.decode('ascii') for id, name in zip(metadata['id'], metadata['name'])}
        msids = list(mdmap.values())

        # Create any msid archive directories that do not already exists
        _create_msid_directories(msids)

        # Create any msid archive files that do not already exists
        _create_archive_files(msids)

        # # Initialize the dataset in the archive for any new msids
        # _create_msid_datasets(msids)

        logger.info(
            f"Preparing to append {len(large_sample)}"
            f" new datapoints to the archive for {len(msids)} msids ..."
        )

        _organize_data_for_append(
            large_sample=large_sample,
            mdmap=mdmap
        )

        logger.info(
            f"Starting to append {len(large_sample)}"
            f" new datapoints to the archive for {len(msids)} msids ..."
        )

        _append_h5_col_tlm(msids)

        processed_files = processed_files + file_processing_chunk

    processing_end_time = Time(Time.now(), format="datetime").iso
    print(f"{processing_start_time} | {processing_end_time}")

    return processed_files


def move_archive_files(filetype, processed_ingest_files):

    if processed_ingest_files is not None:

        import tarfile
        import shutil

        os.chdir(STAGING_DIRECTORY)

        tarfile_name = f"stage_{int(DateTime().secs)}.tar"
        tar = tarfile.open(tarfile_name, mode='w')

        for ingest_file in processed_ingest_files:
            tar.add(ingest_file)
            os.remove(ingest_file)

        tar.close()
        shutil.move(
            tarfile_name,
            f"{msid_files['processed_files_directory'].abs}/{tarfile_name}"
        )


def get_archive_files():
    """Get list of files to ingest by examining the file staging area
    """

    files = []

    logger.info(f"Starting legacy file discovery in {STAGING_DIRECTORY} ... ")
    files.extend(sorted(glob.glob(f"{STAGING_DIRECTORY}E*.{opt.ingest_file_format.lower()}")))
    files.extend(sorted(glob.glob(f"{STAGING_DIRECTORY}E*.{opt.ingest_file_format.upper()}")))

    logger.info(f"{len(files)} file(s) staged in {STAGING_DIRECTORY} ...")

    return files

if __name__ == "__main__":
    main()
