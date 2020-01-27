# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
import os
import shutil
from glob import glob

old = '/proj/sot/ska/data/eng_archive/data'
bak = '/proj/sot/ska/data/eng_archive/data_stats_bak'
new = '/proj/sot/ska/tmp/eng_archive/data'

# For testing
#old = '/home/SOT/git/eng_archive/dataold'
#new = '/home/SOT/git/eng_archive/datanew'
#bak = '/home/SOT/git/eng_archive/databak'

dryrun = True

dirs = sorted(glob(os.path.join(old, '*')))
for dir in dirs:
    content = os.path.basename(dir)
    print 'Content:', content
    bak_content = os.path.join(bak, content)
    old_content = os.path.join(old, content)
    for stat in ('5min', 'daily'):
        bak_stat = os.path.join(bak, content, stat)
        old_stat = os.path.join(old, content, stat)
        new_stat = os.path.join(new, content, stat)
        if not os.path.exists(old_stat):
            print 'Skipping {}/{} not in old_stat'.format(content, stat)
            continue
            
        if not os.path.exists(new_stat):
            print 'Skipping {}/{} not in new_stat'.format(content, stat)
            continue
            
        if os.path.exists(bak_stat):
            print 'Skipping {}/{} already done'.format(content, stat)
            continue

        if not os.path.exists(bak_content):
            print 'Making dir', bak_content
            if not dryrun:
                os.mkdir(bak_content)

        print 'mv {} {}'.format(old_stat, bak_content)
        if not dryrun:
            shutil.move(old_stat, bak_content)
            if os.path.exists(old_stat):
                raise Exception('fail old_stat {} still there'.format(old_stat))
            if not os.path.exists(bak_stat):
                raise Exception('fail bak_stat {} not there'.format(bak_stat))

        print 'mv {} {}'.format(new_stat, old_content)
        if not dryrun:
            shutil.move(new_stat, old_content)
            if os.path.exists(new_stat):
                raise Exception('fail new_stat {} still there'.format(new_stat))
            if not os.path.exists(old_stat):
                raise Exception('fail old_stat {} not there'.format(old_stat))
