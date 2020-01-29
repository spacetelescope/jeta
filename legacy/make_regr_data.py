#! /usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pdb

import re, os , sys
import optparse
import shutil

import tables
import numpy as np
import cPickle as pickle

import asciitable
from Chandra.Time import DateTime
import Ska.engarchive.file_defs as file_defs
import Ska.DBI
import pyyaks.context
import pyyaks.logger

loglevel = pyyaks.logger.VERBOSE
logger = pyyaks.logger.get_logger(name='make_regr_data', level=loglevel, format="%(asctime)s %(message)s")

# Globals related to Ska (flight) eng_archive
# Context dictionary to provide context for msid_files
ft = pyyaks.context.ContextDict('ft')
msid_files = pyyaks.context.ContextDict('msid_files', basedir=file_defs.msid_root)
msid_files.update(file_defs.msid_files)
arch_files = pyyaks.context.ContextDict('arch_files', basedir=file_defs.arch_root)
arch_files.update(file_defs.arch_files)

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--start",
                      default='2010:260',
                      help="Start time")
    parser.add_option("--stop",
                      default='2010:270',
                      help="Stop time")
    parser.add_option("--data-root",
                      default="test_eng_archive",
                      help="Engineering archive root directory for MSID files")
    parser.add_option("--contents",
                      default="acis2eng,acis3eng,acisdeahk,orbitephem0,simcoor,thm1eng,ccdm4eng,dp_acispow128",
                      help="Content type to process (default = all)")
    return parser.parse_args()

def get_interval_files(tstart, tstop):

    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs)

    files = db.fetchall('SELECT * FROM archfiles '
                        'WHERE filetime >= ? AND filetime <= ? '
                        'ORDER BY filetime ASC',
                        (DateTime(tstart).secs, DateTime(tstop).secs))
    return files

def get_arch_filename(arch_files, filename):
    ft['basename'] = os.path.basename(filename)
    tstart = re.search(r'(\d+)', str(ft['basename'])).group(1)
    datestart = DateTime(tstart).date
    ft['year'], ft['doy'] = re.search(r'(\d\d\d\d):(\d\d\d)', datestart).groups()

    return arch_files['archfile'].abs

def copy_archfiles_to_test(file_records):
    """
    Copy all archive ``file_records`` for ``content`` from the flight Ska
    archive directory to a corresponding test archive directory with the same
    structure.
    """
    arch_filenames = [get_arch_filename(arch_files, x['filename'])
                       for x in file_records]
    test_arch_filenames = [get_arch_filename(test_arch_files, x['filename'])
                            for x in file_records]
    for infile, outfile in zip(arch_filenames, test_arch_filenames):
        if not os.path.exists(outfile):
            test_archdir = os.path.dirname(outfile)
            if not os.path.exists(test_archdir):
                logger.info('makedirs %s' % test_archdir)
                os.makedirs(test_archdir)
            logger.info('cp %s %s' % (infile, outfile))
            shutil.copy2(infile, outfile)

def make_test_content_dir():
    contentdir = test_msid_files['contentdir'].abs
    if not os.path.exists(contentdir):
        os.makedirs(contentdir)
    shutil.copy2(msid_files['colnames'].abs, test_msid_files['colnames'].abs)
    shutil.copy2(msid_files['colnames_all'].abs, test_msid_files['colnames_all'].abs)

def make_test_archfiles_db(file_records):
    if os.path.exists(test_msid_files['archfiles'].abs):
        return
    shutil.copy2(msid_files['archfiles'].abs, test_msid_files['archfiles'].abs)
    db = Ska.DBI.DBI(dbi='sqlite', server=test_msid_files['archfiles'].abs)
    db.execute('DELETE FROM archfiles', commit=True)
    rowstart0 = file_records[0]['rowstart']
    for record in file_records:
        record['rowstart'] -= rowstart0
        record['rowstop'] -= rowstart0
        db.insert(record, 'archfiles', commit=False)
    db.commit()
    db.execute('vacuum')

def copy_msidfiles_to_test(file_records, rowstart, rowstop):
    n_rows = rowstop - rowstart
    colnames = pickle.load(open(msid_files['colnames'].abs))
    for colname in colnames:
        ft['msid'] = colname
        if os.path.exists(test_msid_files['data'].abs):
            continue
        logger.info('Copying MSID {0}'.format(colname))
        shutil.copy(msid_files['data'].abs, test_msid_files['data.tmp'].abs)
        h5 = tables.openFile(test_msid_files['data.tmp'].abs, 'a')
        h5.root.data[:n_rows] = h5.root.data[rowstart:rowstop]
        h5.root.quality[:n_rows] = h5.root.quality[rowstart:rowstop]
        h5.root.data.truncate(n_rows)
        h5.root.quality.truncate(n_rows)
        h5.copyFile(test_msid_files['data'].abs, overwrite=True)
        #print h5.root.data[0], h5.root.data[-1], len(h5.root.data), rowstart, rowstop
        h5.close()
        os.unlink(test_msid_files['data.tmp'].abs)

def copy_statfiles_to_test(stat, dt, tstart, tstop):
    ft['interval'] = stat
    colnames = pickle.load(open(msid_files['colnames'].abs))
    for colname in colnames:
        ft['msid'] = colname
        if os.path.exists(test_msid_files['stats'].abs):
            continue
        if os.path.exists(msid_files['stats'].abs):
            logger.info('Copying {0} stats for MSID {1}'.format(stat, colname))
            statdir = os.path.dirname(test_msid_files['stats.tmp'].abs)
            if not os.path.exists(statdir):
                os.makedirs(statdir)
            shutil.copy(msid_files['stats'].abs, test_msid_files['stats.tmp'].abs)
            h5 = tables.openFile(test_msid_files['stats.tmp'].abs, 'a')
            times = (h5.root.data.col('index') + 0.5) * dt
            row0, row1 = np.searchsorted(times, [tstart, tstop])
            #print colname, row0, row1, len(times), DateTime(times[row0]).date, DateTime(times[row1]).date,
            # Remove from row1-1 to end.  The row1-1 is because it is possible
            # to get the daily stat without the rest of the 5min data if
            # tstop is past noon of the day.  This messes up update_archive.
            h5.root.data.removeRows(row1 - 1, h5.root.data.nrows)
            h5.root.data.removeRows(0, row0)
            h5.copyFile(test_msid_files['stats'].abs, overwrite=True)
            newtimes = (h5.root.data.col('index') + 0.5) * dt
            #print len(newtimes), DateTime(newtimes[0]).date, DateTime(newtimes[-1]).date
            h5.close()
            os.unlink(test_msid_files['stats.tmp'].abs)

def make_filetypes_dat(contents):
    filetypes = asciitable.read(msid_files['filetypes'].abs)
    test_filetypes = [x for x in filetypes if x['content'].lower() in contents]
    asciitable.write(test_filetypes, os.path.join(opt.data_root, 'filetypes.dat'),
                     names=filetypes.dtype.names)

opt, args = get_options()

test_msid_files = pyyaks.context.ContextDict('msid_files', basedir=opt.data_root)
test_msid_files.update(file_defs.msid_files)
test_arch_files = pyyaks.context.ContextDict('arch_files', basedir=opt.data_root)
test_arch_files.update(file_defs.arch_files)

contents = [x.lower() for x in opt.contents.split(',')]
if 'all' in contents:
    filetypes = asciitable.read(msid_files['filetypes'].abs)
    contents = [x['content'].lower() for x in filetypes]
    print contents

for content in contents:
    ft['content'] = content
    logger.info("Making content {0}".format(content.upper()))
    make_test_content_dir()
    file_records = get_interval_files(opt.start, opt.stop)
    if not content.lower().startswith('dp_'):
        copy_archfiles_to_test(file_records)

    tstart = file_records[0]['tstart']
    tstop = file_records[-1]['tstop']
    rowstart = file_records[0]['rowstart']
    rowstop = file_records[-1]['rowstop']
    print 'tstart, tstop, datestart, datestop, rowstart, rowstop', (
        tstart, tstop, DateTime(tstart).date, DateTime(tstop).date,
        rowstart, rowstop)

    copy_msidfiles_to_test(file_records, rowstart, rowstop)
    copy_statfiles_to_test('5min', 328, tstart, tstop)
    copy_statfiles_to_test('daily', 86400, tstart, tstop)

    make_test_archfiles_db(file_records)

make_filetypes_dat(contents)

