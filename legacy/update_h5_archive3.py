# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pyfits
import time
import tables
import numpy as np
import Ska.Table
import re
import os
import sys
import glob

def make_h5_col_file(dat, content, colname):
    """Make a new h5 table to hold column from ``dat``."""
    filename = os.path.join('data', content, 'msid', colname + '.h7')
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
                    expectedrows=86400*365*10)
    h5.createEArray(h5.root, 'quality', tables.BoolAtom(), (0,), title='Quality',
                    expectedrows=86400*365*10)
    print 'Made', colname
    h5.close()

def append_h5_col(hdus, content, colname, i_colname):
    filename = os.path.join('data', content, 'msid', colname + '.h7')
    h5 = tables.openFile(filename, mode='a')
    h5.root.data.append(np.hstack([x[1].field(colname) for x in hdus]))
    h5.root.quality.append(np.hstack([x[1].field('QUALITY')[:,i_colname] for x in hdus]))
    h5.close()

filetypes = Ska.Table.read_ascii_table('filetypes.dat')
filetypes = filetypes[ filetypes.pipe == 'ENG0' ]

for filetype in filetypes:
    if filetype.content != 'PCAD3ENG':
        continue
    print filetype
    content = filetype.content.lower()
    instrum = filetype.instrum.lower()
    fitsdir = os.path.abspath(os.path.join('data', content, 'fits'))

    fitsfiles = sorted(glob.glob(os.path.join(fitsdir, '*.fits.gz')))
    if not fitsfiles:
        continue

    dat = Ska.Table.read_fits_table(fitsfiles[0])
    for colname in dat.dtype.names:
        make_h5_col_file(dat, content, colname)

    h5dir = os.path.join('data', content, 'msid')
    if not os.path.exists(h5dir):
        os.makedirs(h5dir)

    hdus = []
    for i, f in enumerate(fitsfiles):
        print 'Reading', i, len(fitsfiles), f
        hdus.append(pyfits.open(f))

    for i_colname, colname in enumerate(dat.dtype.names):
        if colname != 'QUALITY':
            print '.',
            append_h5_col(hdus, content, colname, i_colname)
        print

