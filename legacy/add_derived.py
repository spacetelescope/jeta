#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re, os, sys
import cPickle as pickle
import optparse

import tables
from Chandra.Time import DateTime

import Ska.DBI
import pyyaks.logger
import pyyaks.context
import numpy as np

import Ska.engarchive.fetch as fetch 
import Ska.engarchive.file_defs as file_defs
import Ska.engarchive.derived as derived

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--data-root",
                      default=".",
                      help="Engineering archive root directory for MSID and arch files")
    parser.add_option("--start",
                      default="1999:201",
                      help="Start for initial data fetch")
    parser.add_option("--stop",
                      default="1999:260",
                      help="Stop for initial data fetch")
    parser.add_option("--content",
                      action='append',
                      help="Content type to process [match regex] (default = all)")
    return parser.parse_args()

def make_archfiles_db(filename, content_def):
    # Do nothing if it is already there
    if os.path.exists(filename):
        return

    datestart = DateTime(DateTime(opt.start).secs - 60)
    tstart = datestart.secs
    tstop = tstart
    year, doy = datestart.date.split(':')[:2]
    times, indexes = derived.times_indexes(tstart, tstop, content_def['time_step'])

    logger.info('Creating db {}'.format(filename))
    archfiles_def = open('archfiles_def.sql').read()
    db = Ska.DBI.DBI(dbi='sqlite', server=filename)
    db.execute(archfiles_def)
    archfiles_row = dict(filename='{}:0:1'.format(content_def['content']),
                         filetime=0,
                         year=year,
                         doy=doy,
                         tstart=tstart,
                         tstop=tstop,
                         rowstart=0,
                         rowstop=0,
                         startmjf=indexes[0], # really index0
                         stopmjf=indexes[-1],  # really index1
                         date=datestart.date)
    db.insert(archfiles_row, 'archfiles')

def add_colname(filename, colname):
    """Add ``colname`` to the pickled set() in ``filename``.  Create the pickle
    as needed.
    """
    if not os.path.exists(filename):
        logger.info('Creating colnames pickle {}'.format(filename))
        with open(filename, 'w') as f:
            pickle.dump(set(), f)

    colnames = pickle.load(open(filename, 'r'))
    if colname not in colnames:
        logger.info('Adding colname {} to colnames pickle {}'.format(colname, filename))
        colnames.add(colname)
        with open(filename, 'w') as f:
            pickle.dump(colnames, f)
    
def make_msid_file(colname, content, content_def):
    ft['content'] = content
    ft['msid'] = colname
    filename = msid_files['data'].abs
    if os.path.exists(filename):
        return

    logger.info('Making MSID data file %s', filename)

    if colname == 'TIME':
        dp_vals, indexes = derived.times_indexes(opt.start, opt.stop,
                                                 content_def['time_step'])
    else:
        dp = content_def['classes'][colname]()
        dataset = dp.fetch(opt.start, opt.stop)
        dp_vals = np.asarray(dp.calc(dataset), dtype=dp.dtype)

    # Finally make the actual MSID data file
    filters = tables.Filters(complevel=5, complib='zlib')
    h5 = tables.openFile(filename, mode='w', filters=filters)
    
    n_rows = int(20 * 3e7 / content_def['time_step'])
    h5shape = (0,) 
    h5type = tables.Atom.from_dtype(dp_vals.dtype)
    h5.createEArray(h5.root, 'data', h5type, h5shape, title=colname,
                    expectedrows=n_rows)
    h5.createEArray(h5.root, 'quality', tables.BoolAtom(), (0,), title='Quality',
                    expectedrows=n_rows)

    logger.info('Made {} shape={} with n_rows(1e6)={}'.format(colname, h5shape, n_rows / 1.0e6))
    h5.close()

def main():
    global opt, ft, msid_files, logger

    opt, args = get_options()
    ft = fetch.ft
    msid_files = pyyaks.context.ContextDict('add_derived.msid_files', basedir=opt.data_root)
    msid_files.update(file_defs.msid_files)
    logger = pyyaks.logger.get_logger(name='engarchive', level=pyyaks.logger.VERBOSE, 
                                      format="%(asctime)s %(message)s")

    # Get the derived parameter classes
    dp_classes = (getattr(derived, x) for x in dir(derived) if x.startswith('DP_'))
    dp_classes = [x for x in dp_classes if hasattr(x, '__base__') and
                                           issubclass(x, derived.DerivedParameter)]
    content_defs = {}
    for dp_class in dp_classes:
        colname = dp_class.__name__.upper()
        dp = dp_class()
        content = dp.content
        if opt.content == [] or any(re.match(x + r'\d+', content) for x in opt.content):
            dpd = content_defs.setdefault(content, {})
            dpd.setdefault('classes', {'TIME': None})
            dpd['content'] = content
            dpd['classes'][colname] = dp_class
            dpd['mnf_step'] = dp.mnf_step
            dpd['time_step'] = dp.time_step

    for content, content_def in content_defs.items():
        ft['content'] = content
        logger.info('CONTENT = {}'.format(content))

        # Make content directory
        if not os.path.exists(msid_files['contentdir'].rel):
            logger.info('Making directory {}'.format(msid_files['contentdir'].rel))
            os.mkdir(msid_files['contentdir'].rel)

        # Make the archfiles.db3 file (if needed)
        make_archfiles_db(msid_files['archfiles'].abs, content_def)

        for colname in content_def['classes']:
            ft['msid'] = colname
            logger.debug('MSID = {}'.format(colname))
            # Create colnames and colnames_all pickle files (if needed) and add colname
            add_colname(msid_files['colnames'].rel, colname)
            add_colname(msid_files['colnames_all'].rel, colname)

            make_msid_file(colname, content, content_def)

        add_colname(msid_files['colnames_all'].rel, 'QUALITY')

if __name__ == '__main__':
    main()
    
