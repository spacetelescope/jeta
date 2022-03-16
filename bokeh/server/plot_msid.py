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

from bokeh.models.widgets import RadioButtonGroup

class PlotServer:

    def generate_full_res_plot(self):
       
        data = fetch.MSID(
            msid=self.msid, 
            start=Time(self.tstart, format='datetime').yday, 
            stop=Time(self.tstop, format='datetime').yday
        )

        x_axis = Time(data.times, format='unix').datetime
        y_axis = data.vals

        source = ColumnDataSource(dict(x=x_axis, y=y_axis))

        main_line = self.plot.step(
            x_axis, 
            y_axis, 
            mode='after', 
            color='royalblue', 
            width=2, name=self.msid, 
            legend_label='raw telemetry'
        )
        self.plot.circle(
            x_axis, 
            y_axis, 
            fill_color='white', 
            line_color='royalblue', 
            size=5, 
            legend_label='raw telemetry'
        )

        hover_tool = HoverTool(tooltips=[
                    ('value', "$y"),
                    ("obs. time", "@x{%Y:%j:%H:%M:%S.%3N}"),
                ],
                formatters={
                    '@x': 'datetime',
                    '$y' : 'printf',
                },
        )
        self.plot.tools.append(hover_tool)
        self.plot.legend.click_policy="hide"

        return self.plot

    def generate_5min_stats_plot(self):

        # Fetch 5min telemetry 
        data = fetch.MSID(
            msid=self.msid, 
            start=Time(self.tstart, format='datetime').yday, 
            stop=Time(self.tstop, format='datetime').yday, 
            stat='5min'
        )
    
        x_axis = Time(data.times, format='unix').datetime

        source = ColumnDataSource(data=dict(
            x=x_axis, 
            y=data.means, 
            mins=data.mins, 
            maxes=data.maxes, 
            means=data.means
        ))

        min_line = self.plot.line(
            x_axis, 
            data.mins, 
            color="green", 
            line_dash='dotdash', 
            line_width=1, 
            name='min',
            legend_label='min'
        )
        max_line = self.plot.line(
            x_axis, 
            data.maxes, 
            color="blue", 
            line_dash='dotdash', 
            line_width=1, 
            name='max',
            legend_label='max'
        )
        main_line = self.plot.line(
            x_axis, 
            data.means, 
            color='black', 
            width=2, 
            name=self.msid, 
            legend_label='mean'
        )
       
        hover_tool = HoverTool(tooltips=[
                    ('value', "$y"),
                    ("obs. time", "@x{%Y:%j:%H:%M:%S.%3N}"),
                ],
                formatters={
                    '@x': 'datetime',
                    '$y' : 'printf',
                },
        )
        self.plot.tools.append(hover_tool)
        self.plot.legend.click_policy="hide"

        return self.plot

    def selection_resolution_interval(self):
        assert self.tstop > self.tstart, 'tstop must come after tstart'
        diff_interval = self.tstop - self.tstart
        if abs(diff_interval).days > 365 * 5:
            interval = 'daily'
        elif abs(diff_interval).days > 1:
            interval = '5min'
        else:
            interval = 'full'
        return interval

    def serve_plot(self):
        {
            'full': self.generate_full_res_plot, 
            'daily': self.generate_full_res_plot, 
            '5min': self.generate_5min_stats_plot
        }[self.interval]()
        curdoc().add_root(self.plot)
        
    def __init__(self, args=curdoc().session_context.request.arguments):
        self.msid = args.get('msid')[0].decode('utf-8')

        self.msid_time_range = fetch.get_time_range(self.msid, 'iso')
        self.msid_min_time = Time(self.msid_time_range[0], format='iso').yday
        self.msid_max_time = Time(self.msid_time_range[1], format='iso').yday
        
        self.tstart = args.get('tstart')[0].decode('utf-8')
        self.tstart = Time(self.tstart if self.tstart else self.msid_min_time, format='yday').datetime
        self.tstop = args.get('tstop')[0].decode('utf-8').replace('/', '')
        self.tstop = Time(self.tstop if self.tstop else self.msid_max_time, format='yday').datetime

        self.interval = self.selection_resolution_interval()
        self.xaxis_formatter = DatetimeTickFormatter(  microseconds = ['%H:%M:%S.%3N'],
                                                    milliseconds = ['%H:%M:%S.%3N'],
                                                    seconds = ['%H:%M:%S'],
                                                    minsec = ['%Y:%j:%H:%M:%S'],
                                                    minutes = ['%Y:%j:%H:%M'],
                                                    hourmin = ['%Y:%j:%H:%M'],
                                                    hours = ['%Y:%j:%H:%M'],
                                                    days = ['%Y:%j:%H:%M'],
                                                    months = ['%Y:%j'],
                                                    years = ['%Y'])
        self.plot = figure(
            title=f"Telemetry Trending Using {self.interval.capitalize()} Interval Selection", 
            x_axis_label='Observatory Time', 
            y_axis_label=f'{self.msid}', 
            x_axis_type='datetime',
            sizing_mode='stretch_width',
            plot_height=400,
            output_backend='webgl'
        )
        self.plot.xaxis.formatter = self.xaxis_formatter
        self.serve_plot()

PlotServer()