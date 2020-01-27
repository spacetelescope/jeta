#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Rename MSID, 5min and daily h5 files based on a set of MSID aliases.
Also updates colnames.pickle and colnames_all.pickle.
"""

import os
import argparse
import pickle

import pyyaks.context

import Ska.engarchive.fetch as fetch
from Ska.engarchive.converters import ALIASES
import Ska.engarchive.file_defs as file_defs


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Dry run (no actual file or database updatees)")
    parser.add_argument("--data-root",
                        default=".",
                        help="Engineering archive root directory for MSID and arch files")
    parser.add_argument("--content",
                        default="sim_mrg",
                        help="Content type to process (default='sim_mrg')")
    return parser.parse_args(args)

opt = get_options()

ft = fetch.ft
ft['content'] = opt.content.lower()
msid_files = pyyaks.context.ContextDict('msid_files',
                                        basedir=opt.data_root)
msid_files.update(file_defs.msid_files)

colnames = [x for x in pickle.load(open(msid_files['colnames'].abs))]

aliases = ALIASES[opt.content]

for colname in (x for x in colnames if x in aliases):
    for filetype, interval in (('msid', ''),
                               ('stats', '5min'),
                               ('stats', 'daily')):
        ft['interval'] = interval
        ft['msid'] = colname
        oldname = msid_files[filetype].rel
        ft['msid'] = aliases[colname]
        newname = msid_files[filetype].rel
        if os.path.exists(oldname):
            print 'mv {} {}'.format(oldname, newname)
            if not opt.dry_run:
                os.rename(oldname, newname)
        else:
            print 'No file for {} ({} does not exist)'.format(colname, oldname)

colnames = pickle.load(open(msid_files['colnames'].abs))
colnames = set(aliases.get(x, x) for x in colnames)
colnames_all = pickle.load(open(msid_files['colnames_all'].abs))
colnames_all = set(aliases.get(x, x) for x in colnames_all)

print 'Updating {} and {}'.format(msid_files['colnames'].rel,
                                  msid_files['colnames_all'].rel)
if not opt.dry_run:
    with open(msid_files['colnames'].rel, 'w') as f:
        pickle.dump(colnames, f)
    with open(msid_files['colnames_all'].rel, 'w') as f:
        pickle.dump(colnames_all, f)
