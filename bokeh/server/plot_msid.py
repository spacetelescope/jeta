# msid_full_resolution_plot.py

# Plotting Libraries
from bokeh.core.property.numeric import Interval
from bokeh.io.doc import curdoc
from bokeh.models.annotations import Legend, LegendItem
# from bokeh.models.callbacks import CustomJS, Div
from bokeh.models.ranges import Range1d
from bokeh.models.widgets.groups import CheckboxGroup
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Step, renderers, Line
from bokeh.models import DatetimeTickFormatter, HoverTool, PointDrawTool, ColorBar
from bokeh.client import push_session
from bokeh.layouts import row
from bokeh import events
from bokeh.models import Button, CustomJS, Div
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6    
from bokeh.layouts import row


# jeta packages
from jeta.archive import fetch
from astropy.time import Time
from numpy import empty

args = curdoc().session_context.request.arguments

# User supplied parameters
MSID = args.get('msid')[0].decode('utf-8')
tstart = args.get('tstart')[0].decode('utf-8')
tstop = args.get('tstop')[0].decode('utf-8').replace('/', '')

print('=-=-=-=-=-=CLIENT-=-=-=-=--=-=-')
print(tstart)
print(tstop)

earliest_tstart, latest_tstop = fetch.get_time_range(MSID, 'date')

print('=-=-=-=-=JETA=-=-=-=--=-=-')
print(earliest_tstart)
print(latest_tstop)

if tstart == '':
    tstart = earliest_tstart

if tstop == '':
    tstop = latest_tstop

_tstart = Time(tstart, format='yday').datetime
_tstop = Time(tstop, format='yday').datetime

print('=-=-=-=-=-=BoKeh-=-=-=-=--=-=-')
print(tstart)
print(tstop)

from bokeh.models.widgets import RadioButtonGroup


def display_event(div, attributes=[], style = 'float:left;clear:left;font_size=13px'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div), code="""
        const attrs = %s;
        const args = [];
        for (let i = 0; i<attrs.length; i++) {
            args.push(attrs[i] + '=' + Number(cb_obj[attrs[i]]).toFixed(2));
        }
        const line = "<span style=%r><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";
        console.log(line);
        const text = div.text.concat(line);
        console.log(text);
        const lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))


def selection_resolution_interval(_tstart=_tstart, _tstop=_tstop):
    diff_interval = _tstop - _tstart

    if abs(diff_interval).days > 365 * 10:
        interval = 'daily'
    elif abs(diff_interval).days > 1:
        interval = '5min'
    else:
        interval = 'full'
    return interval

def generate_5min_stats_plot(msid=MSID, tstart=tstart, tstop=tstop):
    # prepare some data
    p = figure(
        title=f"MSID: {str(MSID).upper()} | Plot Resolution: {selection_resolution_interval(_tstart, _tstop)}", 
        x_axis_label='observatory time', 
        y_axis_label=f'{MSID}', 
        x_axis_type='datetime',
        sizing_mode='stretch_width',
        plot_height=400,
        output_backend='webgl'
    )

    # create a new plot with a title and axis labels
    p.xaxis.formatter = DatetimeTickFormatter(  microseconds = ['%H:%M:%S.%3N'],
                                                milliseconds = ['%H:%M:%S.%3N'],
                                                seconds = ['%H:%M:%S'],
                                                minsec = ['%Y:%j:%H:%M:%S'],
                                                minutes = ['%Y:%j:%H:%M'],
                                                hourmin = ['%Y:%j:%H:%M'],
                                                hours = ['%Y:%j:%H:%M'],
                                                days = ['%Y:%j:%H:%M'],
                                                months = ['%Y:%j'],
                                                years = ['%Y'])

    # Min/Mean/Max (5min)
    msid = fetch.MSID(MSID, tstart, tstop, stat='5min')
  
    x = Time(msid.times, format='unix').datetime

    mins = msid.mins
    maxes = msid.maxes
    means = msid.means

    y = msid.means

    source = ColumnDataSource(data=dict(x=x, y=y, mins=mins, maxes=maxes, means=means))

    min_line = p.line(
        x, 
        mins, 
        color="green", 
        line_dash='dotdash', 
        line_width=1, 
        name='min',
        legend_label='min'
    )
    # p.circle(
    #     x, 
    #     mins, 
    #     fill_color='green', 
    #     size=5,
    #     legend_label='min'
    # )
    max_line = p.line(
        x, 
        maxes, 
        color="blue", 
        line_dash='dotdash', 
        line_width=1, 
        name='max',
        legend_label='max'
    )
    # p.circle(
    #     x, 
    #     maxes, 
    #     fill_color='blue', 
    #     size=5,
    #     legend_label='max'
    # )
    main_line = p.line(x, y, color='black', width=2, name=MSID, legend_label='mean')
    # p.circle(x, y, fill_color='white', line_color='black', size=5, legend_label='mean')

    hover_tool = HoverTool(tooltips=[
                ('value', "$y"),
                ("obs. time", "@x{%Y:%j:%H:%M:%S.%3N}"),
            ],
            formatters={
                '@x': 'datetime',
                '$y' : 'printf',
            },
    )
    p.tools.append(hover_tool)
    p.legend.click_policy="hide"

    def five_update(attr, old, new):
        # callback
        # min_line.visible = False #0 in checkbox.active
        print(MSID)
        msid = fetch.MSID(MSID, tstart, tstop, stat='5min')
        
        x = Time(msid.times, format='unix').datetime

        mins = msid.mins
        maxes = msid.maxes
        means = msid.means
        import numpy as np
        y = np.linspace(0, 4*np.pi, len(means))

        source.data = dict(x=x, y=y, mins=mins, maxes=maxes, means=means)

    # max_line.visible = False #2 in checkbox.active
    # p.legend.items = [legend_items[i] for i in checkbox.active]
    data_selection = RadioButtonGroup(
        labels = ["mean", "midval", "value"],
    active = 0)
    data_selection.on_change('active', five_update)
    layout = row(data_selection, p)
    p = layout
    return p


def generate_full_res_plot(msid=MSID, tstart=tstart, tstop=tstop):
    # prepare some data
    p = figure(
        title=f"MSID: {str(MSID).upper()} | Plot Resolution: {selection_resolution_interval(_tstart, _tstop)}", 
        x_axis_label='observatory time', 
        y_axis_label=f'{MSID}', 
        x_axis_type='datetime',
        sizing_mode='stretch_width',
        plot_height=400,
        output_backend='webgl'
    )

    # create a new plot with a title and axis labels
    p.xaxis.formatter = DatetimeTickFormatter(  microseconds = ['%H:%M:%S.%3N'],
                                                milliseconds = ['%H:%M:%S.%3N'],
                                                seconds = ['%H:%M:%S'],
                                                minsec = ['%Y:%j:%H:%M:%S'],
                                                minutes = ['%Y:%j:%H:%M'],
                                                hourmin = ['%Y:%j:%H:%M'],
                                                hours = ['%Y:%j:%H:%M'],
                                                days = ['%Y:%j:%H:%M'],
                                                months = ['%Y:%j'],
                                                years = ['%Y'])

    msid = fetch.MSID(MSID, tstart, tstop)

    x = Time(msid.times, format='unix').datetime
    y = msid.vals

    source = ColumnDataSource(dict(x=x, y=y))

    main_line = p.step(x, y, mode='after', color='royalblue', width=2, name=MSID, legend_label=f'{MSID}')
    p.circle(x, y, fill_color='white', line_color='royalblue', size=5, legend_label=f'{MSID}')

    hover_tool = HoverTool(tooltips=[
                ('value', "$y"),
                ("obs. time", "@x{%Y:%j:%H:%M:%S.%3N}"),
            ],
            formatters={
                '@x': 'datetime',
                '$y' : 'printf',
            },
    )
    p.tools.append(hover_tool)
    p.legend.click_policy="hide"

    return p

if selection_resolution_interval() == '5min':
    p = generate_5min_stats_plot()
if selection_resolution_interval() == 'full':
    p = generate_full_res_plot()
curdoc().add_root(p)
