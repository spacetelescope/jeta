# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Apply a scale factor of 36.0/35.0 to account for a mistake in the original specification
of the OBC_AVE thermal derived parameter.

This is done as part of https://github.com/sot/eng_archive/pull/127, but original discussion
and code is in https://github.com/sot/eng_archive/pull/115

To start::

  rsync -av --exclude=arch /proj/sot/ska/data/eng_archive/data/dp_thermal128/ data/dp_thermal128/

Then::

  python fix_oba_ave.py
"""

import tables

scale = 36. / 35.

h5 = tables.openFile('data/dp_thermal128/DP_OBA_AVE.h5', 'a')
val = h5.root.data
val[:] = val[:] * scale
h5.close()

h5 = tables.openFile('data/dp_thermal128/5min/DP_OBA_AVE.h5', 'a')
for attr in 'val min max mean'.split():
    val = getattr(h5.root.data.cols, attr)
    val[:] = val[:] * scale
h5.close()

h5 = tables.openFile('data/dp_thermal128/daily/DP_OBA_AVE.h5', 'a')
for attr in 'val min max mean std p01 p05 p16 p50 p84 p95 p99'.split():
    val = getattr(h5.root.data.cols, attr)
    val[:] = val[:] * scale
h5.close()
