# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pyyaks.context import ContextDict

basedir = 'test'

FT = ContextDict('ft')
ft = FT.accessor()

FILES = ContextDict('files', basedir=basedir)
FILES.update({'contentdir':   '{{ft.content}}/',
              'headers':      '{{ft.content}}/headers.pickle',
              'columns':      '{{ft.content}}/columns.pickle',
              'columns_all':  '{{ft.content}}/columns_all.pickle',
              'msiddir':      '{{ft.content}}/msid/',
              'msid':         '{{ft.content}}/msid/{{ft.msid}}',
              'archdir':      '{{ft.content}}/arch/{{ft.year}}/{{ft.doy}}/',
              'archfile':     '{{ft.content}}/arch/{{ft.year}}/{{ft.doy}}/{{ft.basename}}',
              })
files = FILES.accessor()

# Original archive files
OFILES = ContextDict('of', basedir='/data/cosmos2/tlm')
OFILES.update({'contentdir':   '{{ft.content}}/',
           })
ofiles = OFILES.accessor()
