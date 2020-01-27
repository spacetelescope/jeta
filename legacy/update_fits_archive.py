# Licensed under a 3-clause BSD style license - see LICENSE.rst
import time
import re, os , sys
import glob

import numpy as np
import pexpect

import Ska.Table
import arc5gl
from Chandra.Time import DateTime

filetypes = Ska.Table.read_ascii_table('filetypes.dat')
filetypes = filetypes[ filetypes.pipe == 'ENG0' ]

datestop = DateTime(time.time(), format='unix').date

FITS_dir = 'test/tlm'

FT = ContextDict('ft')
ft = FT.accessor()

FILES = ContextDict('files', basedir=fits_dir)
FILES.update({'outdir': '{{ft.content}}',
              'image': '{{ft_id}}/image'
              })
files = FILES.accessor()

loglevel = pyyaks.logger.INFO
logfile = 'run.log'
logger = pyyaks.logger.get_logger(level=loglevel, filename=logfile, format="%(message)s")


for filetype in filetypes:
    try:
        ft.content = filetype.content.lower()
        ft.instrum = filetype.instrum.lower()

        if os.path.exists(outdir):
            print 'Skipping', content, 'at', time.ctime()
            continue
        else:
            os.makedirs(outdir)

        os.chdir(outdir)
        arc5 = arc5gl.Arc5gl()

        print '**********', content, time.ctime(), '***********'
        print '  cd ' + outdir
        arc5.sendline('cd ' + outdir)

        for year in range(2000, 2010):
            if os.path.exists('/pool14/wink/stoptom'):
                raise RuntimeError('Stopped by sherry')
            datestart = '%d:001:00:00:00' % year
            datestop = '%d:001:00:00:00' % (year+1)

            print '  tstart=%s' % datestart
            print '  tstop=%s' % datestop
            print '  get %s_eng_0{%s}' % (instrum, content)

            arc5.sendline('tstart=%s' % datestart)
            arc5.sendline('tstop=%s;' % datestop)
            arc5.sendline('get %s_eng_0{%s}' % (instrum, content))

        open('.process', 'w')

    finally:
        # explicitly close connection to archive
        if 'arc5' in globals():
            del arc5


        
