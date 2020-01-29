# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fix overlaps within the original archive and h5 ingest.  These are due to
problems with different versioned files that cover same time range being
retrieved by the archive.
"""

import os
import shutil
import optparse

import numpy
import tables
import Ska.DBI
import asciitable

from context_def import ft, files


def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--dry-run",
                      action="store_true",
                      help="Dry run (no actual file or database updatees)")
    parser.add_option("--overwrite-partial",
                      action="store_true",
                      help="Overwrite the overlapping bits of file (e.g. for EPHEM)")
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
    if not os.path.exists(files['archfiles'].abs) or os.path.exists(files['oldmsid'].abs + '.bak'):
        print 'Skipping', ft['content']
        continue

    # Make backup!
    print 'Copying ', files['oldmsid'].abs, files['oldmsid'].abs + '.bak'
    if not opt.dry_run:
        shutil.copy2(files['oldmsid'].abs, files['oldmsid'].abs + '.bak')

    # Read the list of archive files that were used to build the MSID h5 files
    db = Ska.DBI.DBI(dbi='sqlite', server=files['archfiles'].abs)
    archfiles = db.fetchall('select * from archfiles order by tstart asc')
    db.conn.close()

    # Find archive files that overlap in time (tstart : tstop).
    overlaps = archfiles[1:].tstart - archfiles[:-1].tstop < -1
    file0s = archfiles[:-1][overlaps]
    file1s = archfiles[1:][overlaps]

    # Open the TIME.h5 file for this content type
    h5 = tables.openFile(files['oldmsid'].abs, mode=('r' if opt.dry_run else 'a'))

    # Iterate through overlapping files and set bad quality in TIME for rows
    # coming from the file with lower revision
    for file0, file1 in zip(file0s, file1s):
        if opt.overwrite_partial:
            # Effectively overwrite the file0 values with file1 values
            # for the file0 rows from file1['tstart'] to file0['tstop']
            times = h5.root.data[file0['rowstart']:file0['rowstop']]
            bad_rowstart = numpy.searchsorted(times, file1['tstart']) + file0['rowstart']
            bad_rowstop = file0['rowstop']
        else:
            badfile = file0 if file0['revision'] < file1['revision'] else file1
            bad_rowstart = badfile['rowstart']
            bad_rowstop = badfile['rowstop']
        print file0['filename'], file0['revision'], file1['revision'], \
            '%9d %9d %9d %9d' % (file0['tstart'], file0['tstop'],
                                 file1['tstart'], file1['tstop']), \
            'Setting TIME rows', bad_rowstart, bad_rowstop, 'to bad quality'
        if not opt.dry_run:
            h5.root.quality[bad_rowstart:bad_rowstop] = True

    h5.close()
