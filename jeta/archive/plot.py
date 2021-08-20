# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import num2epoch, epoch2num

import numpy as np
from astropy.time import Time

from matplotlib.dates import (YearLocator, MonthLocator, DayLocator,
                              HourLocator, MinuteLocator, SecondLocator,
                              DateFormatter, epoch2num)
from matplotlib.ticker import FixedLocator, FixedFormatter


MIN_TSTART_UNIX = Time('1999:100', format='yday').unix
MAX_TSTOP_UNIX = Time(Time.now()).unix + 1e7


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Provide useful utilities for matplotlib."""

# Default tick locator and format specification for making nice time axes
TICKLOCS = ((YearLocator, {'base': 5}, '%Y',    YearLocator, {'base': 1}),
            (YearLocator, {'base': 4}, '%Y',    YearLocator, {'base': 1}),
            (YearLocator, {'base': 2}, '%Y',    YearLocator, {'base': 1}),
            (YearLocator, {'base': 1}, '%Y', MonthLocator, {'bymonth': (1, 4, 7, 10)}),
            (MonthLocator, {'bymonth': list(range(1, 13, 6))}, '%Y-%b', MonthLocator, {}),
            (MonthLocator, {'bymonth': list(range(1, 13, 4))}, '%Y-%b', MonthLocator, {}),
            (MonthLocator, {'bymonth': list(range(1, 13, 3))}, '%Y-%b', MonthLocator, {}),
            (MonthLocator, {'bymonth': list(range(1, 13, 2))}, '%Y-%b', MonthLocator, {}),
            (MonthLocator, {},         '%Y-%b', DayLocator, {'bymonthday': (1, 15)}),

            (DayLocator, {'interval': 10}, '%Y:%j', DayLocator, {}),
            (DayLocator, {'interval': 5}, '%Y:%j', DayLocator, {}),
            (DayLocator, {'interval': 4}, '%Y:%j', DayLocator, {}),
            (DayLocator, {'interval': 2}, '%Y:%j', DayLocator, {}),
            (DayLocator, {'interval': 1}, '%Y:%j', HourLocator, {'byhour': (0, 6, 12, 18)}),

            (HourLocator, {'byhour': list(range(0, 24, 12))}, '%j:%H:00', HourLocator, {}),
            (HourLocator, {'byhour': list(range(0, 24, 6))}, '%j:%H:00', HourLocator, {}),
            (HourLocator, {'byhour': list(range(0, 24, 4))}, '%j:%H:00', HourLocator, {}),
            (HourLocator, {'byhour': list(range(0, 24, 2))}, '%j:%H:00', HourLocator, {}),
            (HourLocator, {}, '%j:%H:00', MinuteLocator, {'byminute': (0, 15, 30, 45)}),

            (MinuteLocator, {'byminute': (0, 30)}, '%j:%H:%M', MinuteLocator, {'byminute': list(range(0,60,5))}),
            (MinuteLocator, {'byminute': (0, 15, 30, 45)}, '%j:%H:%M', MinuteLocator, {'byminute': list(range(0,60,5))}),
            (MinuteLocator, {'byminute': list(range(0, 60, 10))}, '%j:%H:%M', MinuteLocator, {}),
            (MinuteLocator, {'byminute': list(range(0, 60, 5))}, '%j:%H:%M', MinuteLocator, {}),
            (MinuteLocator, {'byminute': list(range(0, 60, 4))}, '%j:%H:%M', MinuteLocator, {}),
            (MinuteLocator, {'byminute': list(range(0, 60, 2))}, '%j:%H:%M', MinuteLocator, {}),
            (MinuteLocator, {}, '%j:%H:%M', SecondLocator, {'bysecond': (0, 15, 30, 45)}),

            (SecondLocator, {'bysecond': (0, 30)}, '%H:%M:%S', SecondLocator, {'bysecond': list(range(0,60,5))}),
            (SecondLocator, {'bysecond': (0, 15, 30, 45)}, '%H:%M:%S', SecondLocator, {'bysecond': list(range(0,60,5))}),
            (SecondLocator, {'bysecond': list(range(0, 60, 10))}, '%H:%M:%S', SecondLocator, {}),
            (SecondLocator, {'bysecond': list(range(0, 60, 5))}, '%H:%M:%S', SecondLocator, {}),
            (SecondLocator, {'bysecond': list(range(0, 60, 4))}, '%H:%M:%S', SecondLocator, {}),
            (SecondLocator, {'bysecond': list(range(0, 60, 2))}, '%H:%M:%S', SecondLocator, {}),
            (SecondLocator, {}, '%H:%M:%S', SecondLocator, {}),
            )


def set_time_ticks(plt, ticklocs=None):
    """
    Pick nice values to show time ticks in a date plot.

    Example::

      x = cxctime2plotdate(np.linspace(0, 3e7, 20))
      y = np.random.normal(size=len(x))

      fig = pylab.figure()
      plt = fig.add_subplot(1, 1, 1)
      plt.plot_date(x, y, fmt='b-')
      ticklocs = set_time_ticks(plt)

      fig.autofmt_xdate()
      fig.show()

    The returned value of ``ticklocs`` can be used in subsequent date plots to
    force the same major and minor tick locations and formatting.  Note also
    the use of the high-level fig.autofmt_xdate() convenience method to configure
    vertically stacked date plot(s) to be well-formatted.

    :param plt: ``matplotlib.axes.AxesSubplot`` object (from ``pylab.figure.add_subplot``)
    :param ticklocs: list of major/minor tick locators ala the default ``TICKLOCS``
    :rtype: tuple with selected ticklocs as first element
    """

    locs = ticklocs or TICKLOCS

    for majorLoc, major_kwargs, major_fmt, minorLoc, minor_kwargs in locs:
        plt.xaxis.set_major_locator(majorLoc(**major_kwargs))
        plt.xaxis.set_minor_locator(minorLoc(**minor_kwargs))
        plt.xaxis.set_major_formatter(DateFormatter(major_fmt))

        majorticklocs = plt.xaxis.get_ticklocs()
        if len(majorticklocs) >= 5:
            break

    return ((majorLoc, major_kwargs, major_fmt, minorLoc, minor_kwargs), )


def remake_ticks(ax):
    """Remake the date ticks for the current plot if space is pressed.  If '0'
    is pressed then set the date ticks to the maximum possible range.
    """
    ticklocs = set_time_ticks(ax)
    ax.figure.canvas.draw()


def plot_cxctime(times, y, fmt='-b', fig=None, ax=None, yerr=None, xerr=None, tz=None,
                 state_codes=None, interactive=True, **kwargs):
    """Make a date plot where the X-axis values are in a CXC time compatible format.  If no ``fig``
    value is supplied then the current figure will be used (and created
    automatically if needed).  If yerr or xerr is supplied, ``errorbar()`` will
    be called and any additional keyword arguments will be passed to it.
    Otherwise any additional keyword arguments (e.g. ``fmt='b-'``) are passed
    through to the ``plot()`` function.  Also see ``errorbar()`` for an
    explanation of the possible forms of *yerr*/*xerr*.

    If the ``state_codes`` keyword argument is provided then the y-axis ticks and
    tick labels will be set accordingly.  The ``state_codes`` value must be a list
    of (raw_count, state_code) tuples, and is normally set to ``msid.state_codes``
    for an MSID object from fetch().

    If the ``interactive`` keyword is True (default) then the plot will be redrawn
    at the end and a GUI callback will be created which allows for on-the-fly
    update of the date tick labels when panning and zooming interactively.  Set
    this to False to improve the speed when making several plots.  This will likely
    require issuing a plt.draw() or fig.canvas.draw() command at the end.

    :param times: CXC time values for x-axis (DateTime compatible format, CxoTime)
    :param y: y values
    :param fmt: plot format (default = '-b')
    :param fig: pyplot figure object (optional)
    :param yerr: error on y values, may be [ scalar | N, Nx1, or 2xN array-like ]
    :param xerr: error on x values in units of DAYS (may be [ scalar | N, Nx1, or 2xN array-like ] )
    :param tz: timezone string
    :param state_codes: list of (raw_count, state_code) tuples
    :param interactive: use plot interactively (default=True, faster if False)
    :param ``**kwargs``: keyword args passed through to ``plot_date()`` or ``errorbar()``

    :rtype: ticklocs, fig, ax = tick locations, figure, and axes object.
    """
    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.gcf()

    if ax is None:
        ax = fig.gca()

    if yerr is not None or xerr is not None:
        ax.errorbar(time2plotdate(times), y, yerr=yerr, xerr=xerr, fmt=fmt, **kwargs)
        ax.xaxis_date(tz)
    else:
        ax.plot_date(time2plotdate(times), y, fmt=fmt, **kwargs)
    ticklocs = set_time_ticks(ax)
    fig.autofmt_xdate()

    if state_codes is not None:
        counts, codes = zip(*state_codes)
        ax.yaxis.set_major_locator(FixedLocator(counts))
        ax.yaxis.set_major_formatter(FixedFormatter(codes))

    # If plotting interactively then show the figure and enable interactive resizing
    if interactive and hasattr(fig, 'show'):
        fig.canvas.draw()
        ax.callbacks.connect('xlim_changed', remake_ticks)

    return ticklocs, fig, ax


def time2plotdate(times):
    """
    Convert input CXC time (sec) to the time base required for the matplotlib
    plot_date function (days since start of year 1)?

    :param times: times (any DateTime compatible format or object)
    :rtype: plot_date times
    """
    # # Convert times to float array of CXC seconds
    # if isinstance(times, (Time, Time)):
    #     times = times.unix
    # else:
    times = np.asarray(times)

    # If not floating point then use CxoTime to convert to seconds
    # if times.dtype.kind != 'f':
    #     times = Time(times).unix

    # Find the plotdate of first time and use a relative offset from there
    t0 = Time(times[0], format='unix').unix
    plotdate0 = epoch2num(t0)
    return (times - times[0]) / 86400. + plotdate0


def pointpair(x, y=None):
    """Interleave and then flatten two arrays ``x`` and ``y``.  This is
    typically useful for making a histogram style plot where ``x`` and ``y``
    are the bin start and stop respectively.  If no value for ``y`` is provided then
    ``x`` is used.

    Example::

      from Ska.Matplotlib import pointpair
      x = np.arange(1, 100, 5)
      x0 = x[:-1]
      x1 = x[1:]
      y = np.random.uniform(len(x0))
      xpp = pointpair(x0, x1)
      ypp = pointpair(y)
      plot(xpp, ypp)

    :x: left edge value of point pairs
    :y: right edge value of point pairs (optional)
    :rtype: np.array of length 2*len(x) == 2*len(y)
    """
    if y is None:
        y = x
    return np.array([x, y]).reshape(-1, order='F')


def hist_outline(dataIn, *args, **kwargs):
    """
    histOutline from http://www.scipy.org/Cookbook/Matplotlib/UnfilledHistograms

    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.

    Example Usage:
    binsIn = np.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    (bins, data) = histOutline(binsIn, angle)
    plot(bins, data, 'k-', linewidth=2)

    """

    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

def get_stat(t0, t1, npix):
    t0 = Time(t0)
    t1 = Time(t1)
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
        self.tstart = Time(num2epoch(x0), format='unix').unix
        self.tstop = Time(num2epoch(x1), format='unix').unix
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
