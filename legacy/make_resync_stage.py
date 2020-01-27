#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Create stage directory structure resync the OCC eng archive with the
HEAD archive.
"""

import os
import shutil
import glob
import Ska.Table
from Chandra.Time import DateTime
import Ska.engarchive.fetch as fetch
import Ska.engarchive.file_defs as file_defs
import optparse
import pyyaks.context

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--dry-run",
                      action="store_true",
                      help="Dry run (no actual file or database updatees)")
    parser.add_option("--date",
                      default="2011:200",
                      help="Starting year:doy for assembling resync archive files")
    parser.add_option("--content",
                      action='append',
                      help="Content type to process (default = all)")
    parser.add_option("--outdir",
                      default="stage",
                      help="Output directory")
    return parser.parse_args()

opt, args = get_options()

msid_files = pyyaks.context.ContextDict('msid_files', file_defs.msid_root)
msid_files.update(file_defs.msid_files)

arch_files = pyyaks.context.ContextDict('arch_files', basedir=file_defs.arch_root)
arch_files.update(file_defs.arch_files)

ft = fetch.ft

date = DateTime(opt.date).date
year0, doy0 = date[0:4], date[5:8]

# Get the archive content filetypes
filetypes = Ska.Table.read_ascii_table(msid_files['filetypes'].abs)
if opt.content:
    contents = [x.upper() for x in opt.content]
    filetypes = [x for x in filetypes if x['content'] in contents]

for filetype in filetypes:
    ft['content'] = filetype['content'].lower()
    print "Copying {} content".format(ft['content'])
    archtime = DateTime("{}:{}".format(year0, doy0)).secs

    outdir = os.path.join(opt.outdir, ft['content'].val)
    if not os.path.exists(outdir):
        if not opt.dry_run:
            os.mkdir(outdir)

    while True:
        date = DateTime(archtime).date
        ft['year'], ft['doy'] = date[0:4], date[5:8]
        print "  {}:{}".format(ft['year'], ft['doy'])
        archdir = arch_files['archdir'].abs
        if os.path.exists(archdir):
            for filename in glob.glob(os.path.join(archdir, '*.fits.gz')):
                if not os.path.exists(os.path.join(outdir, os.path.basename(filename))):
                    # print "Copying {} to {}".format(filename, outdir)
                    if not opt.dry_run:
                        shutil.copy2(filename, outdir)

        archtime += 86400
        if archtime > DateTime().secs:
            break

