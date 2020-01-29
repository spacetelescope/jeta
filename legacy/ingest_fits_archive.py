# Licensed under a 3-clause BSD style license - see LICENSE.rst
import time
import os
import sys

import asciitable
import Ska.arc5gl as arc5gl

filetypes = asciitable.read('Ska/engarchive/filetypes.dat')
if len(sys.argv) >= 2:
    filetypes = filetypes[filetypes['content'] == sys.argv[1].upper()]
outroot = sys.argv[2] if len(sys.argv) >= 3 else '/data/cosmos2/eng_archive/tlm'


for filetype in filetypes:
    if filetype['content'] in ('ORBITEPHEM', 'LUNAREPHEM', 'SOLAREPHEM'):
        doys = range(1, 372, 5)
    else:
        doys = (1, 367)

    try:
        content = filetype['content'].lower()
        arc5gl_query = filetype['arc5gl_query'].lower()

        outdir = os.path.abspath(os.path.join(outroot, content))
        if os.path.exists(outdir):
            print 'Skipping', content, 'at', time.ctime()
            continue
        else:
            print 'Making dir', outdir
            os.makedirs(outdir)

        os.chdir(outdir)
        arc5 = arc5gl.Arc5gl()

        print '**********', content, time.ctime(), '***********'
        print '  cd ' + outdir
        arc5.sendline('cd ' + outdir)
        arc5.sendline('version = last')

        for year in range(1999, 2000):
            if os.path.exists('/pool14/wink/stoptom'):
                raise RuntimeError('Stopped by sherry')
            for doy0, doy1 in zip(doys[:-1], doys[1:]):
                datestart = '%d:%03d:00:00:00' % (year, doy0)
                datestop = '%d:%03d:00:00:00' % (year, doy1)

                sys.stdout.flush()
                print '  tstart=%s' % datestart
                print '  tstop=%s' % datestop
                print '  get %s at %s' % (arc5gl_query, time.ctime())

                arc5.sendline('tstart=%s' % datestart)
                arc5.sendline('tstop=%s;' % datestop)
                arc5.sendline('get %s' % arc5gl_query.lower())

        open('.process', 'w')

    finally:
        # explicitly close connection to archive
        if 'arc5' in globals():
            del arc5
