# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Check that the 5min and daily stats are consistent between two different
builds of the stats archive.
"""
import sys
import optparse
import re
import os
import numpy as np
import tables
import matplotlib.pyplot as plt
from Ska.engarchive import fetch
from Chandra.Time import DateTime

if ('ENG_ARCHIVE' in os.environ
    and os.environ['ENG_ARCHIVE'] != '/proj/sot/ska/data/eng_archive'):
    raise ValueError('ENG_ARCHIVE must be set to /proj/sot/ska/data/eng_archive')

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--stat",
                      default='daily',
                      help="stat to check")
    parser.add_option("--content",
                      action='append',
                      help="Content type to process [match regex] (default = all)")
    return parser.parse_args()

opt, args = get_options()

stat = opt.stat
IMAX = {'5min': 20000, 'daily': 4000}[stat]
DT = {'5min': 328, 'daily': 86400}[stat]

content = fetch.content
content_types = sorted(set(content.values()))
if opt.content:
    contents = [x.lower() for x in opt.content]
    content_types = [x for x in content_types
                     if any(re.match(y, x) for y in contents)]

class NotEnoughRows(ValueError):
    pass

def getstats(filename):
    try:
        h5f = tables.openFile(filename, 'r')
        if len(h5f.root.data) < IMAX:
            raise NotEnoughRows
        means = h5f.root.data.col('mean')[:IMAX]
        stds = h5f.root.data.col('std')[:IMAX]
        vals = h5f.root.data.col('val')[:IMAX]
        mins = h5f.root.data.col('min')[:IMAX]
        maxes = h5f.root.data.col('max')[:IMAX]
        indexes = h5f.root.data.col('index')[:IMAX]
    except:
        raise
    finally:
        h5f.close()
    return means, vals, mins, maxes, indexes, stds

for content_type in reversed(content_types):
    if content_type.startswith('dp_therm') or content_type in ('pcad13eng', 'pcad8eng', 'pcad7eng'):
        # These have known and OK issues.
        # dp_therm* (original flight database) had a repeat of 60 days which causes fail here.
        # The pcad ones are about missing data.
        print 'Skipping', content_type
        continue
    print content_type
    msids = [k for k, v in content.items() if v == content_type]

    for msid in msids:
        # print msid,
        h5f = '/proj/sot/ska/data/eng_archive/data/{}/{}/{}.h5'.format(content[msid], stat, msid.upper())
        h5n = 'data/{}/{}/{}.h5'.format(content[msid], stat, msid.upper())
        try:
            fmeans, fvals, fmins, fmaxes, fidxs, fstds = getstats(h5f)  # flight
            nmeans, nvals, nmins, nmaxes, nidxs, nstds = getstats(h5n)  # new
        except NotEnoughRows:
            print '{}: ERROR - not enough rows'.format(content_type)
            break
        except Exception as err:
            if 'does not have a column named' in str(err):
                continue
            print '{}: ERROR - {}'.format(msid, err)

        idx_mismatch = fidxs != nidxs
        if np.any(idx_mismatch):
            print 'Index mismatch at {}:{} {}'.format(
                DateTime(fidxs[idx_mismatch][0] * DT).date,
                fidxs[idx_mismatch][:3], nidxs[idx_mismatch][:3])
        means_med = abs(np.median(fmeans))
        range = max([means_med, np.max(fmaxes) - np.min(fmins)])
        if range <= 0:
            range = 1.0
        dvals = np.abs(nvals - fvals) / range
        dmeans = np.abs(nmeans - fmeans) / range
        dmaxes = np.abs(nmaxes - fmaxes) / range
        dmins =  np.abs(nmins - fmins) / range
        dstds =  np.abs(nstds - fstds) / range
        viol = False
        for name, vals, limit in (('vals', dvals, 0.01),
                                  ('means', dmeans, 0.01),
                                  ('stds', dstds, 0.01),
                                  ('maxes', dmaxes, 1e-5),
                                  ('mins', dmins, 1e-5)):
            if np.max(vals) > limit:
                if not viol:
                    print '{:14s}'.format(msid),
                    viol = True
                time_of_max = fidxs[np.argmax(vals)] * DT
                date = DateTime(time_of_max).date
                print '{}={:.8f} {}'.format(name, np.max(vals), date),
        if viol:
            print '{} {}'.format(means_med, range)

def plot_stats(msid,
               nmeans, nvals, nmins, nmaxes,
               fmeans, fvals, fmins, fmaxes):
    plt.figure(1, figsize=(10, 8))
    plt.clf()

    plt.subplot(4, 2, 1)
    plt.hist(fvals, bins=50)
    plt.title('vals')

    plt.subplot(4, 2, 2)
    plt.hist(nvals - fvals, bins=50)
    plt.title('delta vals')

    plt.subplot(4, 2, 3)
    plt.hist(fmeans, bins=50)
    plt.title('means')

    plt.subplot(4, 2, 4)
    plt.hist(nmeans - fmeans, bins=50)
    plt.title('delta means')

    plt.subplot(4, 2, 5)
    plt.hist(fmins, bins=50)
    plt.title('mins')

    plt.subplot(4, 2, 6)
    plt.hist(nmins - fmins, bins=50)
    plt.title('delta mins')

    plt.subplot(4, 2, 7)
    plt.hist(fmaxes, bins=50)
    plt.title('maxes')

    plt.subplot(4, 2, 8)
    plt.hist(nmaxes - fmaxes, bins=50)
    plt.title('delta maxes')

    plt.tight_layout()
    plt.savefig('stats_plots/{}.png'.format(msid))
    
