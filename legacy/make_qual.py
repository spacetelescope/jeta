# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Convert boolean quality vectors within each content/MSID.h5 file to an int8
content/MSID_qual.h5 file.  This is a one-off for Matt to be able to use files
in Matlab.
"""
from __future__ import with_statement

import re, os , sys
import glob
import time
import cPickle as pickle

import Ska.File
import Ska.Table

import tables
import numpy as np

from context_def import ft, files

# Get the archive content filetypes
filetypes = Ska.Table.read_ascii_table('filetypes.dat')

for filetype in filetypes:
    ft.val.content = filetype.content.lower()

    if not os.path.exists(files['colnames'].abs):
        continue
    
    colnames = list(pickle.load(open(files['colnames'].abs)))
    for colname in reversed(colnames):
        ft.val.msid = colname
        if os.path.exists(files['qual'].abs):
            continue

        print 'Reading', files['oldmsid'].abs
        h5 = tables.openFile(files['oldmsid'].abs)
        iqual = np.int8(h5.root.quality[:])
        h5.close()

        filters = tables.Filters(complevel=5, complib='zlib')

        print 'Creating', files['qual'].abs
        h5 = tables.openFile(files['qual'].abs, mode='w', filters=filters)
        h5shape = (0,)
        h5type = tables.Atom.from_dtype(iqual.dtype)
        h5.createEArray(h5.root, 'quality', h5type, h5shape, title=colname + ' quality',
                        expectedrows=len(iqual))
        h5.root.quality.append(iqual)
        h5.close()

