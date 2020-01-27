# Licensed under a 3-clause BSD style license - see LICENSE.rst
import tables
import numpy as np
import Ska.Table
import re
import os
import sys
import glob

def get_h5_colname(colname):
    # return colname
    return colname if re.match(r'^[a-zA-Z_]', colname) else '_' + colname

def make_h5_file(dat, filename):
    """Make a new h5 table to hold columns from ``dat``."""
    if os.path.exists(filename):
        os.unlink(filename)
    
    colnames = dat.dtype.names
    filters = tables.Filters(complevel=5, complib='zlib')
    h5 = tables.openFile(filename, mode='w', filters=filters)
    
    for colname in colnames:
        col = dat[colname]
        h5shape = (0,) + col.shape[1:]
        h5colname = get_h5_colname(colname)
        h5type = tables.Atom.from_dtype(col.dtype)
        h5.createEArray(h5.root, h5colname, h5type, h5shape, title=colname,
                        expectedrows=86400*30)
        print 'Made', colname

    return h5

def make_h5_col_file(dat, colname):
    """Make a new h5 table to hold column from ``dat``."""
    filename = 'thm1eng/msid/' + colname + '.h5'
    if os.path.exists(filename):
        os.unlink(filename)
    
    filters = tables.Filters(complevel=5, complib='zlib')
    h5 = tables.openFile(filename, mode='w', filters=filters)
    
    col = dat[colname]
    h5shape = (0,) + col.shape[1:]
    h5colname = get_h5_colname(colname)
    h5type = tables.Atom.from_dtype(col.dtype)
    h5.createEArray(h5.root, h5colname, h5type, h5shape, title=colname,
                    expectedrows=86400*30)
    print 'Made', colname
    h5.close()

def append_h5_col(dat, colname):
    filename = 'thm1eng/msid/' + colname + '.h5'
    h5 = tables.openFile(filename, mode='a')
    h5colname = get_h5_colname(colname)
    h5col = h5.root.__getattr__(h5colname)
    h5col.append(dat[colname])
    h5.close()

dat = Ska.Table.read_fits_table('thm_1_eng0.fits.gz')
for colname in dat.dtype.names:
    make_h5_col_file(dat, colname)

for f in sorted(glob.glob('thm1eng/*.fits.gz')):
    print 'Reading', f
    dat = Ska.Table.read_fits_table(f)

    for colname in dat.dtype.names:
        print '.',
        append_h5_col(dat, colname)
    print

if 0:
    filename = 'test2.h5'
    dat = Ska.Table.read_fits_table('pcadf_proto_3_eng0.fits.gz')
    h5 = make_h5_file(dat, filename)

    for f in glob.glob('pcad3/pcadf*'):
        print 'Reading', f
        dat = Ska.Table.read_fits_table(f)

        for colname in dat.dtype.names:
            h5colname = get_h5_colname(colname)
            h5col = h5.root.__getattr__(h5colname)
            h5col.append(dat[colname])
        h5.flush()
    
