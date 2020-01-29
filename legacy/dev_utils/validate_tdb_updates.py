# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function

import pickle
import os
import Ska.tdb
from Ska.engarchive import fetch_eng as fetch
from matplotlib import pyplot as plt

for content in os.listdir('data'):
    new_colnames = pickle.load(open(os.path.join('data', content, 'colnames.pickle')))
    cur_colnames = pickle.load(open(os.path.join(
                '/proj/sot/ska/data/eng_archive', 'data', content, 'colnames.pickle')))
    print('New {}'.format(content))
    new = new_colnames - cur_colnames
    print(', '.join(sorted(new)))
    lost = cur_colnames - new_colnames
    if lost:
        print('LOST: ', lost)

# Plot representative new vals
d1 = '2016:001'
d2 = '2016:002'

msids = set(['1AHIRADF'])
msids.update(['POLAEV2BT', 'POLINE07T', 'POM2THV1T'])
msids.update(['OHRTHR35_WIDE', '1OLORADF', '1OHIRADF', '2OLORADF', '2OHIRADF', 'OOBTHR30_WIDE'])
msids.update(['AOACIIRS', 'AOACISPX', 'AOACIDPX', 'AOACIMSS'])
msids.update(['4OAVOBAT_WIDE', '4OAVHRMT_WIDE'])
msids.update(['TFSSHDT1', 'TFSSHDT2'])

for msid in msids:
    m = Ska.tdb.msids[msid]
    print(m)
    dat = fetch.Msid(msid, d1, d2)
    plt.figure()
    dat.plot()
    plt.title(msid)

