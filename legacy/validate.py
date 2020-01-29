# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Find gaps (positive and negative) in telemetry times.  Positive gaps are missing data
while negative gaps are duplicate data.  The latter need to be excised (via bad quality).
"""

import tables
from context_def import datadir, ft, files
from Chandra.Time import DateTime

ft['content'] = 'acis2eng'
ft['msid'] = 'TIME'

h5 = tables.openFile(files['oldmsid'].abs)
qual = h5.root.quality[:]
times = h5.root.data[:][~ qual]

dts = times[1:] - times[:-1]
bad = (dts > 66) | (dts < -1)

for gap_time, dt in zip(times[bad], dts[bad]):
    print '%s %12d %7d' % (DateTime(gap_time).date, gap_time, dt)
    
h5.close()
