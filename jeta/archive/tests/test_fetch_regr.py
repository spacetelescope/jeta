#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Regression test for fetching from the Ska engineering archive.

This reads full-resolution, 5 minute, and daily values from a representative
sample of MSIDs for each content type.

To generate or update the regression dataset::

  # In window 2.  This creates fetch_regr_<ska_test_version>
  # If not on GRETA, set the ENG_ARCHIVE env var to point at the flight archive
  % setenv ENG_ARCHIVE /proj/sot/ska/data/eng_archive

  % skatest
  % cd Ska/engarchive/tests
  % python test_fetch_regr.py

Help::

  ska-ccosmos$ ./test_fetch_regr.py --help
  usage: test_fetch_regr.py [-h] [--n-msids N_MSIDS] [--n-samp N_SAMP]
                            [--start START] [--outroot OUTROOT]

  Regression test for fetch

  optional arguments:
    -h, --help         show this help message and exit
    --n-msids N_MSIDS  Max msids per content type
    --n-samp N_SAMP    Max samples of each msid
    --start START      Start
    --outroot OUTROOT  Output file root

"""
from __future__ import print_function, division, absolute_import

import sys
import os
import hashlib
import argparse
import collections

import six
import pytest
from Chandra.Time import DateTime
import numpy as np

WINDOWS = os.name == 'nt'


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Regression test for fetch')
    parser.add_argument('--n-msids',
                        default=4,
                        type=int,
                        help='Max msids per content type')

    parser.add_argument('--n-samp',
                        default=10000,
                        type=int,
                        help='Max samples of each msid')

    parser.add_argument('--start',
                        type=str,
                        default='2012:149:10:00:00',
                        help='Start')

    parser.add_argument('--outfile',
                        type=str,
                        default='fetch_regr.dat',
                        help='Output file (default=stdout)')

    args = parser.parse_args(args)
    return args


def get_contents(fetch, args):
    contents = collections.defaultdict(list)
    for msid, content_type in six.iteritems(fetch.content):
        contents[content_type].append(msid)

    for content_type, msids in contents.items():
        msids = sorted(msids)
        idxs = np.linspace(0, len(msids) - 1, args.n_msids).astype(int)
        contents[content_type] = [msids[idx] for idx in sorted(set(idxs))]

    return contents


def get_md5(fetch, args, msid, start, days, stat):
    md5 = hashlib.md5()
    dat = fetch.MSID(msid, start, start + days, stat=stat)
    n_samp = min(args.n_samp, len(dat.vals))
    idxs = np.linspace(0, n_samp - 1, n_samp).astype(int)
    for val in dat.vals.take(idxs):
        val = '{:.6e}'.format(val) if isinstance(val, np.inexact) else repr(val)
        md5.update(val)
    if dat.bads is not None:
        for bad in dat.bads.take(idxs):
            md5.update(repr(bad))

    return md5.hexdigest()


def main():
    args = get_args()

    from Ska.engarchive import fetch

    contents = get_contents(fetch, args)

    if args.outfile:
        print("Writing regression results to {}".format(args.outfile))
        fout = open(args.outfile, 'w')
    else:
        fout = sys.stdout

    start = DateTime(args.start)
    for i, content_type in enumerate(sorted(contents)):
        msids = contents[content_type]
        for msid in msids:
            print('{} / {} : {} {}'.format(i, len(contents), content_type, msid))
            for stat, days in ((None, 1),
                               ('5min', 4),
                               ('daily', 300)):

                key = '{:20s} {:16s} {:6s}'.format(content_type, msid, stat)
                md5_hex = get_md5(fetch, args, msid, start, days, stat)
                print(key, md5_hex, file=fout)

    if args.outfile:
        fout.close()


def assert_true(x):
    assert x


@pytest.mark.skipif(True, reason='Skip extended regression tests (needs fixing)')
def test_fetch_regr():
    from .. import fetch
    args = get_args(args=[])

    contents = get_contents(fetch, args)
    start = DateTime(args.start)

    regr_md5_hexes = collections.defaultdict(dict)
    outfile = os.path.join(os.path.dirname(__file__), args.outfile)
    with open(outfile, 'r') as f:
        for line in f:
            content_type, msid, stat, md5_hex = line.strip().split()
            key = '{:16s} {:6s}'.format(msid, stat)
            regr_md5_hexes[content_type][key] = md5_hex

    # Should have same content_type keys
    assert sorted(contents) == sorted(regr_md5_hexes)

    print()
    for ii, content_type in enumerate(sorted(contents)):
        test_md5_hexes = {}
        msids = contents[content_type]
        for msid in msids:
            for stat, days in ((None, 1),
                               ('5min', 4),
                               ('daily', 300)):

                key = '{:16s} {:6s}'.format(msid, stat)
                md5_hex = get_md5(fetch, args, msid, start, days, stat)
                test_md5_hexes[key] = md5_hex
        md5s_equal = test_md5_hexes == regr_md5_hexes[content_type]
        sys.stdout.write('Checking {:16s} ({}/{})\r'.
                         format(content_type, ii + 1, len(contents)))
        if not md5s_equal:
            from pprint import pformat
            sys.stdout.write('\nBAD match\n'
                             'Regr:\n'
                             '{}\n'
                             'Test:\n'
                             '{}\n'.format(pformat(regr_md5_hexes[content_type]),
                                           pformat(test_md5_hexes)))
        sys.stdout.flush()
        assert md5s_equal
    print()

if __name__ == '__main__':
    main()
