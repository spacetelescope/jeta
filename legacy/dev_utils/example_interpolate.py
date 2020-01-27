# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from Ska.Matplotlib import plot_cxctime
from Ska.engarchive import fetch_eng as fetch


def plot_both(x, title_str):
    for msid, color in (('aosares1', 'b'), ('pitch_fss', 'r')):
        plot_cxctime(x[msid].times, x[msid].vals, '-o' + color)
        bads = x[msid].bads
        if bads is not None and len(np.flatnonzero(bads)) > 0:
            plot_cxctime(x[msid].times[bads], x[msid].vals[bads], 'kx', ms=10., mew=3)
    title(title_str)


dat = fetch.MSIDset(['aosares1','pitch_fss'], '2010:001:00:00:00','2010:001:00:00:20')
dat['aosares1'].vals = (np.arange(len(dat['aosares1'])) * 4.0)[::-1] + 0.5
dat['aosares1'].bads[2] = True
dat['pitch_fss'].vals = np.arange(len(dat['pitch_fss']))
dat['pitch_fss'].bads[6] = True

def plot_em():
    close('all')
    figure(1, figsize=(6, 4))
    plot_both(dat, 'Original data')
    x0, x1 = xlim()
    dx = (x1 - x0) * 0.05
    x0, x1 = x0 - dx, x1 + dx
    xlim(x0, x1)
    ylim(-0.5, 20)
    grid()
    tight_layout()

    for filter_bad in (False, True):
        for bad_union in (False, True):
            dat2 = dat.interpolate(dt=0.5, filter_bad=filter_bad, bad_union=bad_union, copy=True)
            figure(figsize=(6, 4))
            plot_both(dat2, 'filter_bad={}    bad_union={}'.format(filter_bad, bad_union))
            xlim(x0, x1)
            ylim(-0.5, 20)
            grid()
            tight_layout()
