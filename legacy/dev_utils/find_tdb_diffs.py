# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function
import Ska.tdb

msid_xref = {}
for line in open('/proj/sot/ska/ops/TDB/p014/msids.xref'):
    vals = line.split()
    msid_xref[vals[0]] = vals[1]

versions = (4, 6, 7, 8, 9, 10, 11, 12, 13, 14)
for v1, v2 in zip(versions[:-1], versions[1:]):
    Ska.tdb.set_tdb_version(v1)
    m1 = set(Ska.tdb.msids.Tmsrment['MSID'])

    Ska.tdb.set_tdb_version(v2)
    m2 = set(Ska.tdb.msids.Tmsrment['MSID'])
    print('****** {} vs {} *******'.format(v1, v2))
    if m1 - m2:
        print('** REMOVED **')
        for msid in sorted(m1 - m2):
            print('{:15s}'.format(msid))
    if m2 - m1:
        print('** ADDED **')
        for msid in sorted(m2 - m1):
            print('{:15s} {:15s} {:s}'.format(msid, msid_xref[msid], Ska.tdb.msids[msid].technical_name))
    print()
