# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pyyaks.context import ContextDict

# datadir = '/data/baffin3/telem_archive/data'
datadir = 'data'

ft = ContextDict('ft')

files = ContextDict('files', basedir=datadir)
files.update({'contentdir':   '{{ft.content}}/',
              'headers':      '{{ft.content}}/headers.pickle',
              'colnames':      '{{ft.content}}/colnames.pickle',
              'colnames_all':  '{{ft.content}}/colnames_all.pickle',
              'archfiles':    '{{ft.content}}/archfiles.db3',
              'msiddir':      '{{ft.content}}/msid/',
              'msid':         '{{ft.content}}/msid/{{ft.msid}}.h5',
              'qual':         '{{ft.content}}/{{ft.msid}}_qual.h5',
              'oldmsid':      '{{ft.content}}/{{ft.msid}}.h5',
              # 'oldmsid':      'test.h5',
              'archdir':      '{{ft.content}}/arch/{{ft.year}}/{{ft.doy}}/',
              'archfile':     '{{ft.content}}/arch/{{ft.year}}/{{ft.doy}}/{{ft.basename}}',
              })

# Original archive files
ofiles = ContextDict('of', basedir='/data/cosmos2/tlm')
ofiles.update({'contentdir':   '{{ft.content}}/',
           })

