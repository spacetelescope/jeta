# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Print start or start/stop times for MSID content types.
"""
import glob
import tables

from Chandra.Time import DateTime

if False:
    time_files = glob.glob('/proj/sot/ska/data/eng_archive/data/*/TIME.h5')
    for time_file in time_files:
        h5 = tables.openFile(time_file)
        time0 = h5.root.data[0]
        h5.close()
        print(DateTime(time0).date, time_file)

if True:
    time_files = glob.glob('/proj/sot/ska/data/eng_archive/1999/data/*/TIME.h5')
    for time_file in time_files:
        h5 = tables.openFile(time_file)
        dat = h5.root.data
        if len(dat) > 0:
            time0, time1 = dat[0], dat[-1]
            print(DateTime([time0, time1]).date, time_file)
        else:
            print('EMPTY', time_file)
        h5.close()
