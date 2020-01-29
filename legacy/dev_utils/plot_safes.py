# Licensed under a 3-clause BSD style license - see LICENSE.rst
def plot_pitch(event, stat=None):
    figure()
    start = DateTime(event.start)
    stop = DateTime(event.stop)
    dat = fetch.Msid('pitch', start - 0.5, stop + 1.5, stat=stat)
    if len(dat) > 2:
        dat.plot('b')
    dat = fetch.Msid('aosares1', start - 0.5, stop + 1.5, stat=stat)
    dat.plot('r')
    plot_cxctime([start.secs, start.secs], ylim(), '--r')
    plot_cxctime([stop.secs, stop.secs], ylim(), '--r')
