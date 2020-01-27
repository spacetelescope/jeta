# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function
import os
import pickle

root = '/proj/sot/ska/data/eng_archive/data'
content_dirs = os.listdir(root)
for content_dir in content_dirs:
    f1 = os.path.join(root, content_dir, 'colnames.pickle')
    f2 = os.path.join(root, content_dir, 'colnames_all.pickle')
    if os.path.exists(f1) and os.path.exists(f2):
        colnames = pickle.load(open(f1))
        colnames_all = pickle.load(open(f2))
        diff = colnames_all - colnames - set(['QUALITY'])
        if diff:
            print(content_dir)
            print(diff)
            print()
