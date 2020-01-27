# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Convert dict of archive files and associated info from headers.pickle to
sqlite3 'archfiles' table.  This is a one-off after ingest_h5_archive.py
to correct an original hasty decision about storing file info.
"""
import re, os , sys
import cPickle as pickle
import collections
import sqlite3

import Ska.Table
import Ska.DBI
from context_def import datadir, ft, files

from Chandra.Time import DateTime
import asciitable

filetypes = asciitable.read('Ska/engarchive/filetypes.dat')
if len(sys.argv) == 2:
    filetypes = filetypes[ filetypes['content'] == sys.argv[1].upper() ]

cols_map = dict(TSTART =   'tstart',    
                TSTOP =    'tstop',     
                row0 = 'rowstart',  
                row1 =  'rowstop',   
                STARTMJF = 'startmjf',  
                STARTMNF = 'startmnf',  
                STOPMJF =  'stopmjf',   
                STOPMNF =  'stopmnf',   
                CHECKSUM = 'checksum',  
                TLMVER =   'tlmver',    
                ASCDSVER = 'ascdsver',  
                REVISION = 'revision',  
                DATE =     'date')

archfiles_def = open('archfiles_def.sql').read()

for filetype in filetypes:
    ft['content'] = filetype['content'].lower()

    if not os.path.exists(files['contentdir'].abs) or os.path.exists(files['archfiles'].abs):
        print 'Skipping', ft['content'].val, files['contentdir'].abs, files['archfiles'].abs
        continue

    print 'Processing', ft['content'].val
 
    print 'Creating db', files['archfiles'].abs
    db = Ska.DBI.DBI(dbi='sqlite', server=files['archfiles'].abs, autocommit=False)
    db.execute(archfiles_def)
    db.commit()
    
    print 'Reading', files['headers'].abs
    headers = pickle.load(open(files['headers'].abs))

    
    filetime_counts = collections.defaultdict(int)
    
    n = len(headers)
    for i, filename in enumerate(sorted(headers)):
        header = headers[filename]
        out = dict((outkey, header.get(inkey)) for inkey, outkey in cols_map.items())
        out['filename'] = filename
        filetime = int(re.search(r'(\d+)', filename).group(1))
        out['filetime'] = filetime
        filedate = DateTime(filetime).date
        out['year'], out['doy'] = re.search(r'(\d\d\d\d):(\d\d\d)', filedate).groups()
        
        if i % 100 == 0:
            print i, n, filetime, filename, header['row0'], header['row1'], '\r',
        
        filetime_counts[filetime] += 1
        db.insert(out, 'archfiles')

    db.commit()

    print
    print 'Repeats', [(k, v) for k, v in filetime_counts.items() if v > 1]
    
