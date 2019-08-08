#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re
import os
import optparse
from six.moves import cPickle as pickle

import tables
import pyyaks.logger
import pyyaks.context

import Ska.engarchive.fetch as fetch
import Ska.engarchive.file_defs as file_defs
import Ska.DBI

opt = None
ft = fetch.ft
msid_files = fetch.msid_files
logger = None


def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--data-root",
                      default=".",
                      help="Engineering archive root directory for "
                           "MSID and arch files")
    parser.add_option("--check-order",
                      default=False,
                      action="store_true",
                      help="Check the order of archfiles (increasing in time)")
    parser.add_option("--check-lengths",
                      default=False,
                      action="store_true",
                      help="Check all MSID file lengths")
    parser.add_option("--find-glitch",
                      default=False,
                      action="store_true",
                      help="Find inconsistency in archfiles")
    parser.add_option("--verbose",
                      default=False,
                      action="store_true",
                      help="Verbose")
    parser.add_option("--max-tstart-mismatch",
                      default=100,
                      help="Max mismatch in time between archfiles and h5")
    parser.add_option("--content",
                      action='append',
                      help="Content type to process [match regex] "
                           "(default = all)")
    return parser.parse_args()


def check_filetype(filetype):
    ft['content'] = filetype.content.lower()

    if not os.path.exists(msid_files['archfiles'].abs):
        logger.info('No archfiles.db3 for %s - skipping' % ft['content'])
        return

    logger.info('Checking {} content type, archfiles {}'.format(
        ft['content'], msid_files['archfiles'].abs))

    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs)
    archfiles = db.fetchall('select * from archfiles')
    db.conn.close()

    if opt.check_order:
        for archfile0, archfile1 in zip(archfiles[:-1], archfiles[1:]):
            exception = (archfile0['startmjf'] == 77826 and
                         archfile0['year'] == 2004 and archfile0['doy'] == 309)
            if archfile1['tstart'] < archfile0['tstart'] and not exception:
                logger.info('ERROR: archfile order inconsistency\n {}\n{}'
                            .format(archfile0, archfile1))

    if not opt.check_lengths:
        colnames = ['TIME']
    else:
        colnames = [x for x in pickle.load(open(msid_files['colnames'].abs))
                    if x not in fetch.IGNORE_COLNAMES]

    lengths = set()
    for colname in colnames:
        ft['msid'] = colname

        h5 = tables.openFile(msid_files['msid'].abs, mode='r')
        length = len(h5.root.data)
        h5.root.data[length - 1]
        h5.close()

        logger.verbose('MSID {} has length {}'.format(colname, length))
        lengths.add(length)
        if len(lengths) != 1:
            logger.info('ERROR: inconsistent MSID length {} {} {}'.format(
                ft['content'], colname, lengths))
            return  # Other checks don't make sense now

    length = lengths.pop()

    archfile = archfiles[-1]
    if archfile['rowstop'] != length:
        logger.info('ERROR: inconsistent archfile {}: '
                    'last rowstop={} MSID length={}'.format(
            ft['content'], archfile['rowstop'], length))
        if opt.find_glitch:
            find_glitch()


def find_glitch():
    ft['msid'] = 'TIME'
    h5 = tables.openFile(msid_files['msid'].abs, mode='r')
    times = h5.root.data

    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs)
    archfiles = db.fetchall('select * from archfiles')
    db.conn.close()

    for archfile in archfiles:
        logger.verbose('archfile {} {} {}'.format(
                archfile['filename'], archfile['year'], archfile['doy']))
        tstart = archfile['tstart']
        rowstart = archfile['rowstart']
        if abs(tstart - times[rowstart]) > opt.max_tstart_mismatch:
            logger.info('ERROR: inconsistency\n {}'.format(archfile))
            break

    h5.close()


def main():
    global opt
    global ft
    global msid_files
    global logger

    opt, args = get_options()

    # Set up fetch so it will first try to read from opt.data_root if that is
    # provided as an option and exists, and if not fall back to the default of
    # fetch.ENG_ARCHIVE.  Fetch is a read-only process so this is safe when
    # testing.
    if opt.data_root:
        fetch.msid_files.basedir = ':'.join([opt.data_root, fetch.ENG_ARCHIVE])

    # Set up logging
    loglevel = pyyaks.logger.VERBOSE if opt.verbose else pyyaks.logger.INFO
    logger = pyyaks.logger.get_logger(name='engarchive', level=loglevel,
                                      format="%(asctime)s %(message)s")

    logger.info('Run time options: \n{}'.format(opt))
    logger.info('check_archfiles file: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch module file: {}'
                .format(os.path.abspath(fetch.__file__)))

    # Get the archive content filetypes
    filetypes = fetch.filetypes
    if opt.content:
        contents = [x.upper() for x in opt.content]
        filetypes = [x for x in filetypes
                     if any(re.match(y, x.content) for y in contents)]

    for filetype in filetypes:
        check_filetype(filetype)
