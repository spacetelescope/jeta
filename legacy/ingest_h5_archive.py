# Licensed under a 3-clause BSD style license - see LICENSE.rst
import time
import tables
import pyfits
import numpy as np
import Ska.Table
import asciitable
import re
import os
import sys
import glob
import cPickle as pickle

import Ska.engarchive.converters as converters

def make_h5_col_file(dat, content, colname, n_rows):
    """Make a new h5 table to hold column from ``dat``."""
    filename = os.path.join('data', content, colname + '.h5')
    if os.path.exists(filename):
        os.unlink(filename)
    filedir = os.path.dirname(filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    filters = tables.Filters(complevel=5, complib='zlib')
    h5 = tables.openFile(filename, mode='w', filters=filters)
    
    col = dat[colname]
    h5shape = (0,) + col.shape[1:]
    h5type = tables.Atom.from_dtype(col.dtype)
    h5.createEArray(h5.root, 'data', h5type, h5shape, title=colname,
                    expectedrows=n_rows)
    h5.createEArray(h5.root, 'quality', tables.BoolAtom(), (0,), title='Quality',
                    expectedrows=n_rows)
    print 'Made', colname, 'shape=', h5shape, 'with n_rows(1e6) =', n_rows / 1.0e6
    h5.close()

def append_h5_col(dats, content, colname):
    def i_colname(dat):
        return list(dat.dtype.names).index(colname)
    filename = os.path.join('data', content, colname + '.h5')
    h5 = tables.openFile(filename, mode='a')
    newdata = np.hstack([x[colname] for x in dats])
    print 'Appending to', colname, 'with shape', newdata.shape
    h5.root.data.append(newdata)
    h5.root.quality.append(np.hstack([x['QUALITY'][:,i_colname(x)] for x in dats]))
    h5.close()

filetypes = asciitable.read('Ska/engarchive/filetypes.dat')
if len(sys.argv) >= 2:
    filetypes = filetypes[filetypes['content'] == sys.argv[1].upper()]
outroot = sys.argv[2] if len(sys.argv) >= 3 else '/data/cosmos2/eng_archive/tlm'

for filetype in filetypes:
    content = filetype['content'].lower()
    fitsdir = os.path.abspath(os.path.join(outroot, content))

    if os.path.exists(os.path.join('data', content)):
        print "Skipping", filetype
        continue
    print filetype

    # If files are already in the final cxc archive location:
    # fitsfiles = sorted(glob.glob('/data/cosmos2/eng_archive/data/acisdeahk/arch/????/???/*.fits.gz'))
    fitsfiles = sorted(glob.glob(os.path.join(fitsdir, filetype['fileglob'])))
    if not fitsfiles:
        print 'No files'
        continue

    dat = Ska.Table.read_fits_table(fitsfiles[-1])
    dat = converters.convert(dat, filetype['content'])
    dt = np.median(dat['TIME'][1:] - dat['TIME'][:-1])
    print 'dt=',dt
    n_rows = int(86400 * 365 * 12 / dt)
    colnames = set(dat.dtype.names)
    colnames_all = set(dat.dtype.names)
    for colname in colnames_all:
        if len(dat[colname].shape) > 1:
            print 'Removing column', colname
            colnames.remove(colname)
    for colname in colnames:
        make_h5_col_file(dat, content, colname, n_rows)

    headers = dict()
    max_size = 1e8
    dats_size = 0
    dats = []
    row = 0

    # fitsfiles = fitsfiles[:50]   # for testing a few
    for i, f in enumerate(fitsfiles):
        print 'Reading', i, len(fitsfiles), f
        sys.stdout.flush()
        hdus = pyfits.open(f)
        hdu = hdus[1]
        try:
            dat = converters.convert(hdu.data, filetype['content'])
        except converters.NoValidDataError:
            print 'WARNING: skipping because of no valid data'
            continue
        header = dict((x, hdu.header[x]) for x in hdu.header.keys() if not re.match(r'T.+\d+', x))
        header['row0'] = row
        header['row1'] = row + len(dat)
        if dat['QUALITY'].shape[1] != len(dat.dtype.names):
            print 'WARNING: skipping because of quality size mismatch: ', \
                  dat['QUALITY'].shape[1], len(dat.dtype.names)
            hdus.close()
            continue
        headers[os.path.basename(f)] = header
        dats.append(dat)
        del hdu
        hdus.close()

        colnames_all.update(dat.dtype.names)
        colnames.intersection_update(dat.dtype.names)

        row += len(dat)
        dats_size += dat.itemsize * len(dat)
        if dats_size > max_size or i == len(fitsfiles) - 1:
            print 'Writing accumulated column data to h5 file at', time.ctime()
            sys.stdout.flush()
            for colname in colnames:
                append_h5_col(dats, content, colname)
                sys.stdout.flush()
            print
            dats = []
            dats_size = 0

            # Not really necessary to write each time, but helps in case of problems
            print 'Writing pickle files'
            pickle.dump(headers, open(os.path.join('data', content, 'headers.pickle'), 'w'), protocol=2)
            pickle.dump(colnames, open(os.path.join('data', content, 'colnames.pickle'), 'w'))
            pickle.dump(colnames_all, open(os.path.join('data', content, 'colnames_all.pickle'), 'w'))
            print 'Done writing pickle files'
            sys.stdout.flush()
