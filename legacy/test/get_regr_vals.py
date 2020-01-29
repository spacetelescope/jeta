#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import os
import optparse
import cPickle as pickle

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--test",
                      action="store_true",
                      help="Test")
    parser.add_option("--start",
                      default='2010:273',
                      help="Start time")
    parser.add_option("--stop",
                      default='2010:285',
                      help="Stop time")
    parser.add_option("--out",
                      default='regr_vals',
                      help="Output prefix")
    return parser.parse_args()

opt, args = get_options()
if opt.test:
    sys.path.insert(0, '..')
    if not 'ENG_ARCHIVE' in os.environ:
        os.environ['ENG_ARCHIVE'] = os.path.abspath(os.getcwd() + '/eng_archive')
    outfile = opt.out + '.test'
else:
    outfile = opt.out + '.flight'

import Ska.engarchive.fetch as fetch
print 'Fetch file is', fetch.__file__
print 'ENG_ARCHIVE is', os.environ.get('ENG_ARCHIVE')

msids = ('1crat', 'fptemp_11', 'orbitephem0_x', 'sim_z', 'tephin', 'cvcductr', 'dp_dpa_power')
attrs = ('times', 'vals', 'quals', 'stds', 'mins', 'maxes', 'means',
         'p01s', 'p05s', 'p16s', 'p50s', 'p84s', 'p95s', 'p99s')
out = dict()
for msid in msids:
    print 'Getting', msid
    dat = fetch.MSID(msid, opt.start, opt.stop)
    dat5 = fetch.MSID(msid, opt.start, opt.stop, stat='5min')
    datd = fetch.MSID(msid, opt.start, opt.stop, stat='daily')
    out[msid] = dict(dat=dict((x, getattr(dat, x)) for x in attrs if hasattr(dat, x)),
                     dat5=dict((x, getattr(dat5, x)) for x in attrs if hasattr(dat5, x)),
                     datd=dict((x, getattr(datd, x)) for x in attrs if hasattr(datd, x)))

out['ENG_ARCHIVE'] = os.environ.get('ENG_ARCHIVE')
out['file'] = fetch.__file__
pickle.dump(out, open(outfile, 'w'), protocol=-1)


                     
