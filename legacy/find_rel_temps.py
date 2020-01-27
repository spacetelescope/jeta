# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pickle
from Ska.engarchive import fetch

dat = asciitable.read('/proj/sot/ska/ops/TDB/tmsrment.txt',
                      Reader=asciitable.NoHeader, delimiter=",",
                      quotechar='"')

descrs = dict((msid.upper(), descr) for msid, descr, unit in zip(dat['col1'], dat['col2'], dat['col5'])
             if unit)

cxcunits = pickle.load(open('cxcunits.pkl'))

temp_msids = [msid for msid, unit in cxcunits.items() if unit == 'K']

reltemps = []
for msid in sorted(temp_msids):
    try:
        dat = fetch.MSID(msid, '2010:001', '2010:002', stat='daily')
    except ValueError:
        print msid, 'not in archive'
        continue

    if dat.means[0] < 100:
        print msid, dat.means[0], descrs[msid]
        reltemps.append(msid)
    
