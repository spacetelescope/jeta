#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
One-off tool to move archive files from original flat structure (everything in
one directory per content-type) to <content>/arch/<year>/<doy>/[files] structure.
"""
import re, os , sys
import glob
import time
import shutil

from Chandra.Time import DateTime
import Ska.Table
import pyyaks.logger
import pyyaks.context

import Ska.engarchive.file_defs as file_defs

arch_files = pyyaks.context.ContextDict('arch_files', basedir=file_defs.arch_root)
arch_files.update(file_defs.arch_files)

orig_arch_files = pyyaks.context.ContextDict('orig_arch_files', basedir=file_defs.orig_arch_root)
orig_arch_files.update(file_defs.orig_arch_files)

ft = pyyaks.context.ContextDict('ft')

def main():
    filetypes = Ska.Table.read_ascii_table('filetypes.dat')
    if len(sys.argv) == 2:
        filetypes = filetypes[ filetypes['content'] == sys.argv[1].upper() ]

    loglevel = pyyaks.logger.INFO
    logger = pyyaks.logger.get_logger(level=loglevel, format="%(message)s")

    for filetype in filetypes:
        ft.content = filetype.content.lower()

        orig_files_glob = os.path.join(orig_arch_files['contentdir'].abs, filetype['fileglob'])
        logger.info('orig_files_glob=%s', orig_files_glob)
        for f in glob.glob(orig_files_glob):
            ft.basename = os.path.basename(f)
            tstart = re.search(r'(\d+)', ft.basename).group(1)
            datestart = DateTime(tstart).date
            ft.year, ft.doy = re.search(r'(\d\d\d\d):(\d\d\d)', datestart).groups()

            archdir = arch_files['archdir'].abs
            archfile = arch_files['archfile'].abs

            if not os.path.exists(archdir):
                print 'Making dir', archdir
                os.makedirs(archdir)
                
            if not os.path.exists(archfile):
                # logger.info('mv %s %s' % (f, archfile))
                shutil.move(f, archfile)

if __name__ == '__main__':
    main()
