#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import os
import optparse
import cPickle as pickle
import numpy as np

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--root",
                      default='regr_vals',
                      help="Input prefix")
    return parser.parse_args()

opt, args = get_options()

flight = pickle.load(open(opt.root + '.flight'))
test = pickle.load(open(opt.root + '.test'))

msids = ('1crat', 'fptemp_11', 'orbitephem0_x', 'sim_z', 'tephin', 'cvcductr', 'dp_dpa_power')
attrs = ('times', 'vals', 'quals', 'stds', 'mins', 'maxes', 'means',
         'p01s', 'p05s', 'p16s', 'p50s', 'p84s', 'p95s', 'p99s')

for msid in msids:
    for stat in ('dat', 'dat5', 'datd'):
        for attr in attrs:
            try:
                f = flight[msid][stat]
                t = test[msid][stat]
            except KeyError:
                print 'MSID={} stat={} missing in flight or test data'.format(msid, stat)
                continue
            if attr not in f:
                continue
            print 'Checking', msid, stat, attr,
            if len(f[attr]) != len(t[attr]):
                print 'Length mismatch:', len(f[attr]), len(t[attr])
                continue
            if attr == 'quals':
                ok = f['quals'] == t['quals']
            else:
                ok = np.max(np.abs(f[attr] - t[attr])) < 1e-6
            print ('OK' if ok else 'NOT OK')
                     
