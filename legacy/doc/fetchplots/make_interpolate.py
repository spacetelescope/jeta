# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make plots illustrating MSIDset.interpolate()

% skatest (or skadev)
$ export ENG_ARCHIVE=/proj/sot/ska/data/eng_archive
$ ipython --classic
>>> run -i make_interpolate
>>> plot_em()
"""

import numpy as np

from Ska.Matplotlib import plot_cxctime
from Ska.engarchive import fetch_eng as fetch
import matplotlib.pyplot as plt


dat = fetch.MSIDset(['aosares1', 'pitch_fss'], '2010:001:00:00:00', '2010:001:00:00:20')
dat['aosares1'].vals = (np.arange(len(dat['aosares1'])) * 4.0)[::-1] + 0.5
dat['aosares1'].bads[2] = True
dat['pitch_fss'].vals = np.arange(len(dat['pitch_fss']))
dat['pitch_fss'].bads[6] = True


def plot_both(x, title_str):
    for msid, color in (('aosares1', 'b'), ('pitch_fss', 'r')):
        plot_cxctime(x[msid].times, x[msid].vals, '-o' + color)
        bads = x[msid].bads
        if bads is not None and len(np.flatnonzero(bads)) > 0:
            plot_cxctime(x[msid].times[bads], x[msid].vals[bads], 'kx', ms=10., mew=2.5)
    plt.title(title_str)


def plot_em():
    plt.close('all')
    plt.figure(1, figsize=(6, 4))
    plot_both(dat, 'Original data')
    x0, x1 = plt.xlim()
    dx = (x1 - x0) * 0.05
    x0, x1 = x0 - dx, x1 + dx

    def fix_plot():
        plt.xlim(x0, x1)
        plt.ylim(-0.5, 20)
        plt.grid()
        plt.tight_layout()

    fix_plot()
    plt.savefig('interpolate_input.png')

    for filter_bad in (False, True):
        for bad_union in (False, True):
            dat2 = dat.interpolate(dt=0.5, filter_bad=filter_bad, bad_union=bad_union, copy=True)
            plt.figure(figsize=(6, 4))
            plot_both(dat2, 'filter_bad={}    bad_union={}'.format(filter_bad, bad_union))
            fix_plot()
            plt.savefig('interpolate_{}_{}.png'.format(filter_bad, bad_union))


if __name__ == '__main__':
    plot_em()
