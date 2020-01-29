# Licensed under a 3-clause BSD style license - see LICENSE.rst
import tables
import numpy as np
import Ska.Table
import re

if 'tlm' not in globals():
    print 'Reading tlm'
    tlm = Ska.Table.read_fits_table('/proj/sot/ska/analysis/psmc/fitting/telem_08.fits')
cols = tlm.dtype.names

filters = tables.Filters(complevel=9, complib='zlib')
f32 = tables.Float32Atom()
fileh = tables.openFile('%s.h5' % col, mode='w', filters=filters)

for col in cols:
    # Use ``a`` as the object type for the enlargeable array.
    h5col = col if re.match(r'^[a-zA-Z_]', col) else '_' + col
    array_f = fileh.createEArray(fileh.root, h5col, f32, (0,), title=col)
    array_f.append(tlm[col])
    array_f.flush()
    print 'made and appended', col

fileh.close()

if 0:
    # Use ``a`` as the object type for the enlargeable array.
    array_f = fileh.createEArray(fileh.root, 'array_f', f32, (0,), "Floats")
    print 'created file'
    nxx = 100
    for i in range(nxx):
        print i
        nx = 1000000
        x = 2 * np.pi * np.arange(nx) / nx
        a = np.int32(np.sin(x) * 2.5 + np.random.normal(scale=0.05, size=len(x)))
        print 'made a'

        array_f.append(a)
        array_f.flush()
        print 'appended a'

    # Close the file.
    fileh.close()
    print 'closed file'

    if 0:
        fileh = tables.openFile('earray1.h5')
        print np.mean(fileh.root.array_f)
        fileh.close()


