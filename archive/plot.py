# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import num2epoch, epoch2num

from Ska.Matplotlib import plot_cxctime
from Chandra.Time import DateTime

from .version import version as __version__

MIN_TSTART_UNIX = DateTime('1999:100').unix
MAX_TSTOP_UNIX = DateTime().unix + 1e7


def get_stat(t0, t1, npix):
    t0 = DateTime(t0)
    t1 = DateTime(t1)
    dt_days = t1 - t0

    if dt_days > npix:
        stat = 'daily'
    elif dt_days * (24 * 60 / 5) > npix:
        stat = '5min'
    else:
        stat = None
    return stat


class MsidPlot(object):
    """Make an interactive plot for exploring the MSID data.

    This method opens a new plot figure (or clears the current figure)
    and plots the MSID ``vals`` versus ``times``.  This plot can be
    panned or zoomed arbitrarily and the data values will be fetched
    from the archive as needed.  Depending on the time scale, ``iplot``
    will display either full resolution, 5-minute, or daily values.
    For 5-minute and daily values the min and max values are also
    plotted.

    Once the plot is displayed and the window is selected by clicking in
    it, the plot limits can be controlled by the usual methods (window
    selection, pan / zoom).  In addition following key commands are
    recognized::

      a: autoscale for full data range in x and y
      m: toggle plotting of min/max values
      p: pan at cursor x
      y: toggle autoscaling of y-axis
      z: zoom at cursor x
      ?: print help

    Example::

      dat = fetch.Msid('aoattqt1', '2011:001', '2012:001', stat='5min')
      iplot = Ska.engarchive.MsidPlot(dat)

    Caveat: the ``MsidPlot()`` class is not meant for use within scripts, and
    may give unexpected results if used in combination with other plotting
    commands directed at the same plot figure.

    :param msid: MSID object
    :param fmt: plot format for values (default="-b")
    :param fmt_minmax: plot format for mins and maxes (default="-c")
    :param plot_kwargs: additional plotting keyword args

    """

    def __init__(self, msid, fmt='-b', fmt_minmax='-c', **plot_kwargs):
        self.fig = plt.gcf()
        self.fig.clf()
        self.ax = self.fig.gca()
        self.zoom = 4.0
        self.msid = msid
        self.fetch = msid.fetch
        self.fmt = fmt
        self.fmt_minmax = fmt_minmax
        self.plot_kwargs = plot_kwargs
        self.msidname = self.msid.msid
        self.plot_mins = True
        self.tstart = self.msid.times[0]
        self.tstop = self.msid.times[-1]
        self.scaley = True

        # Make sure MSID is sampled at the correct density for initial plot
        stat = get_stat(self.tstart, self.tstop, self.npix)
        if stat != self.msid.stat:
            self.msid = self.fetch.Msid(self.msidname, self.tstart, self.tstop,
                                        stat=stat)

        self.ax.set_autoscale_on(True)
        self.draw_plot()
        self.ax.set_autoscale_on(False)
        plt.grid()
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

    @property
    def npix(self):
        dims = self.ax.axesPatch.get_window_extent().bounds
        return int(dims[2] + 0.5)

    def key_press(self, event):
        if event.key in ['z', 'p'] and event.inaxes:
            x0, x1 = self.ax.get_xlim()
            dx = x1 - x0
            xc = event.xdata
            zoom = self.zoom if event.key == 'p' else 1.0 / self.zoom
            new_x1 = zoom * (x1 - xc) + xc
            new_x0 = new_x1 - zoom * dx
            tstart = max(num2epoch(new_x0), MIN_TSTART_UNIX)
            tstop = min(num2epoch(new_x1), MAX_TSTOP_UNIX)
            new_x0 = epoch2num(tstart)
            new_x1 = epoch2num(tstop)

            self.ax.set_xlim(new_x0, new_x1)
            self.ax.figure.canvas.draw_idle()
        elif event.key == 'm':
            for _ in range(len(self.ax.lines)):
                self.ax.lines.pop()
            self.plot_mins = not self.plot_mins
            print('\nPlotting mins and maxes is {}'.format(
                'enabled' if self.plot_mins else 'disabled'))
            self.draw_plot()
        elif event.key == 'a':
            # self.fig.clf()
            # self.ax = self.fig.gca()
            self.ax.set_autoscale_on(True)
            self.draw_plot()
            self.ax.set_autoscale_on(False)
            self.xlim_changed(None)
        elif event.key == 'y':
            self.scaley = not self.scaley
            print('Autoscaling y axis is {}'.format(
                'enabled' if self.scaley else 'disabled'))
            self.draw_plot()
        elif event.key == '?':
            print("""
Interactive MSID plot keys:

  a: autoscale for full data range in x and y
  m: toggle plotting of min/max values
  p: pan at cursor x
  y: toggle autoscaling of y-axis
  z: zoom at cursor x
  ?: print help
""")

    def xlim_changed(self, event):
        x0, x1 = self.ax.get_xlim()
        self.tstart = DateTime(num2epoch(x0), format='unix').secs
        self.tstop = DateTime(num2epoch(x1), format='unix').secs
        stat = get_stat(self.tstart, self.tstop, self.npix)

        if (self.tstart < self.msid.tstart or
            self.tstop > self.msid.tstop or
            stat != self.msid.stat):
            dt = self.tstop - self.tstart
            self.tstart -= dt / 4
            self.tstop += dt / 4
            self.msid = self.fetch.Msid(self.msidname, self.tstart, self.tstop,
                                        stat=stat)
        self.draw_plot()

    def draw_plot(self):
        msid = self.msid
        for _ in range(len(self.ax.lines)):
            self.ax.lines.pop()

        # Force manual y scaling
        scaley = self.scaley

        if scaley:
            ymin = None
            ymax = None
            ok = ((msid.times >= self.tstart) &
                  (msid.times <= self.tstop))

        try:
            self.ax.callbacks.disconnect(self.xlim_callback)
        except AttributeError:
            pass

        if self.plot_mins and hasattr(self.msid, 'mins'):
            plot_cxctime(msid.times, msid.mins, self.fmt_minmax,
                         ax=self.ax, fig=self.fig, **self.plot_kwargs)
            plot_cxctime(msid.times, msid.maxes, self.fmt_minmax,
                         ax=self.ax, fig=self.fig, **self.plot_kwargs)
            if scaley:
                ymin = np.min(msid.mins[ok])
                ymax = np.max(msid.maxes[ok])

        vals = msid.raw_vals if msid.state_codes else msid.vals
        plot_cxctime(msid.times, vals, self.fmt,
                     ax=self.ax, fig=self.fig,
                     state_codes=msid.state_codes, **self.plot_kwargs)

        if scaley:
            plotvals = vals[ok]
            if ymin is None:
                ymin = np.min(plotvals)
            if ymax is None:
                ymax = np.max(plotvals)
            dy = (ymax - ymin) * 0.05
            if dy == 0.0:
                dy = min(ymin + ymax, 1e-12) * 0.05
            self.ax.set_ylim(ymin - dy, ymax + dy)

        self.ax.set_title('{} {}'.format(msid.MSID, msid.stat or ''))
        if msid.unit:
            self.ax.set_ylabel(msid.unit)

        # Update the image object with our new data and extent
        self.ax.figure.canvas.draw_idle()

        self.xlim_callback = self.ax.callbacks.connect('xlim_changed',
                                                       self.xlim_changed)
