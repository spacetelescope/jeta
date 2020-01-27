# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fix residual overlaps within the original archive and h5 ingest.  These
are short overlaps that for some reason were not fixed by fix_h5_ingest.py
"""

import os
import optparse

import numpy as np
import tables
import asciitable

from context_def import ft, files


def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--dry-run",
                      action="store_true",
                      help="Dry run (no actual file or database updatees)")
    return parser.parse_args()

opt, args = get_options()

filetypes = asciitable.read('Ska/engarchive/filetypes.dat')
if len(args) > 0:
    filetypes = filetypes[filetypes['content'] == args[0].upper()]

for filetype in filetypes:
    # Update attributes of global ContextValue "ft".  This is needed for
    # rendering of "files" ContextValue.
    print filetype.content

    ft['content'] = filetype['content'].lower()
    ft['msid'] = 'TIME'

    # archive files
    if not os.path.exists(files['oldmsid'].abs + '.bak'):
        print 'Skipping', ft['content'], ' because there is no backup from fix_ingest_h5.py'
        continue

    # Open the TIME.h5 file for this content type
    h5 = tables.openFile(files['oldmsid'].abs, mode=('r' if opt.dry_run else 'a'))
    goods = ~h5.root.quality[:]
    times = h5.root.data[:]
    print(len(times))
    times = times[goods]
    print(len(times))
    while True:
        bad_indexes = np.where(np.diff(times) < 0)[0]
        print bad_indexes
        if len(bad_indexes) == 0:
            break
        i0 = bad_indexes[0]
        time0 = times[i0]

        for i1 in xrange(i0 + 1, len(times) - 1):
            if times[i1] < time0:
                print 'Setting row {} = {} to bad quality'.format(i1, times[i1])
                if not opt.dry_run:
                    h5.root.quality[i1] = True
            else:
                break
        break

    h5.close()
