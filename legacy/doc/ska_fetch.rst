=====================
Command-line fetch
=====================

The ``fetch_ska`` application allows use of the Ska engineering archive
without getting into Python or using any scripting.  From a single command
line tool you can access most of the common processing steps associated with
fetching and using telemetry data:

- Fetch a set of MSIDs over a time range, specifying the sampling as
  either full-resolution, 5-minute, or daily data.
- Filter out bad or missing data.
- Interpolate (resample) all MSID values to a common uniformly-spaced time sequence.
- Remove or select time intervals corresponding to specified Kadi event types.
- Change the time format from CXC seconds (seconds since 1998.0) to something more
  convenient like GRETA time.
- Write the MSID telemetry data to a zip file.

Aside from the first two steps (fetching data and filtering bad data), all the steps are
optional.


Getting started
----------------

The very first thing is to get set up to use the Ska environment following the
instructions in the `Ska Analysis Tutorial <tutorial.html#configure>`_.  Assuming that is
done, then you need to enter the Ska environment using the ``ska`` (or ``skatest``)
alias::

  % ska

(In case you don't use linux frequently, the ``%`` is meant to represent the command
prompt, so don't type that).  After doing ``ska`` you should see your prompt change to
include a ``ska-`` prefix.

Getting help
^^^^^^^^^^^^^

You can get help by asking ``ska_fetch`` to print its command line options::

  % ska_fetch --help

  usage: ska_fetch [-h] [--start START] [--stop STOP] [--sampling SAMPLING]
                   [--unit-system UNIT_SYSTEM] [--interpolate-dt INTERPOLATE_DT]
                   [--remove-events REMOVE_EVENTS] [--select-events SELECT_EVENTS]
                   [--time-format TIME_FORMAT] [--outfile OUTFILE] [--quiet]
                   [--max-fetch-Mb MAX_FETCH_MB] [--max-output-Mb MAX_OUTPUT_MB]
                   MSID [MSID ...]

  Fetch telemetry from the Ska engineering archive.

  Examples
  ========

    # Get full-resolution TEPHIN, AOPCADMD for last 30 days, and save as telem.zip
    % ska_fetch --sampling=5min --outfile=telem.zip --time-format=greta TEPHIN AOPCADMD

    # Get daily temps since 2000, removing times within 100000 seconds of safe- or normal- sun
    % ska_fetch --sampling=daily --outfile=tephin.zip \
                --remove-events='safe_suns[pad=100000] | normal_suns[pad=100000]' \
                tephin tcylaft6

    # Get daily IRU-2 temps since 2004, removing known LTT bad times
    % ska_fetch AIRU2BT --start 2004:001 --sampling=daily --outfile=airu2bt.zip \
                --remove-events='ltt_bads[msid="AIRU2BT"]'

  Arguments
  =========

  positional arguments:
    MSID                  MSID to fetch

  optional arguments:
    -h, --help            show this help message and exit
    --start START         Start time for data fetch (default=<stop> - 30 days)
    --stop STOP           Stop time for data fetch (default=NOW)
    --sampling SAMPLING   Data sampling (full|5min|daily) (default=5min)
    --unit-system UNIT_SYSTEM
                          Unit system for data (eng|sci|cxc) (default=eng)
    --interpolate-dt INTERPOLATE_DT
                          Interpolate to uniform time steps (secs, default=None)
    --remove-events REMOVE_EVENTS
                          Remove kadi events expression (default=None)
    --select-events SELECT_EVENTS
                          Select kadi events expression (default=None)
    --time-format TIME_FORMAT
                          Output time format (secs|date|greta|jd|frac_year|...)
    --outfile OUTFILE     Output file name (default=fetch.zip)
    --quiet               Suppress run-time logging output
    --max-fetch-Mb MAX_FETCH_MB
                          Max allowed memory (Mb) for fetching (default=1000)
    --max-output-Mb MAX_OUTPUT_MB
                          Max allowed memory (Mb) for file output (default=20)

Try it out
^^^^^^^^^^^

There are plenty of options but frequently you'll only need a few.  Let's start by
trying the first example provided in the help output::

  % ska_fetch TEPHIN AOPCADMD --start=2013:001 --stop=2013:030 --sampling=5min \
              --time-format=greta --outfile=telem.zip
  Fetching 5min-resolution data for MSIDS=['TEPHIN', 'AOPCADMD']
    from 2013:001:12:00:00.000 to 2013:030:12:00:00.000
  Writing data to telem.zip

That was easy, now let's unzip the archive and see what we got.  First look at the archive contents::

  % unzip -l telem.zip
  Archive:  telem.zip
    Length      Date    Time    Name
  ---------  ---------- -----   ----
     460424  03-06-2014 11:46   TEPHIN.csv
     221559  03-06-2014 11:46   AOPCADMD.csv
  ---------                     -------
     681983                     2 files

Now let's unzip::

  % unzip telem.zip
  Archive:  telem.zip
    inflating: TEPHIN.csv
    inflating: AOPCADMD.csv

The first data file is a comma-separated values file ``TEPHIN.csv``.   This could be
imported into Excel or any number of other applications.  Let's look at the first few
lines of the file with the linux ``head`` command::

  % head TEPHIN.csv
  times,samples,vals,mins,maxes,means,midvals
  2013001.120424816,10,113.798,113.798,113.798,113.798,113.798
  2013001.120952816,10,113.798,113.798,113.798,113.798,113.798
  2013001.121520816,10,113.798,113.798,113.798,113.798,113.798
  2013001.122048816,10,113.798,113.798,113.798,113.798,113.798
  2013001.122616816,10,113.798,113.798,113.798,113.798,113.798
  2013001.123144816,10,113.798,113.798,113.798,113.798,113.798
  2013001.123712816,10,113.798,113.798,113.798,113.798,113.798
  2013001.124240816,10,113.798,113.798,113.798,113.798,113.798
  2013001.124808816,10,113.798,113.798,113.798,113.798,113.798

For the TEPHIN data the column names are mostly straighforward.  For 5-minute or daily
data, the ``vals`` column is the same as the mean.  This is a convience so you can
use ``vals`` for ``full``, ``5min`` and ``daily`` sampling analysis.  The ``midvals``
column represents the telemetered value at exactly the midpoint of the interval.

Now let's examine the AOPCADMD output::

  % head AOPCADMD.csv
  times,samples,vals,raw_vals
  2013001.120424816,320,NPNT,1
  2013001.120952816,320,NPNT,1
  2013001.121520816,320,NPNT,1
  2013001.122048816,320,NPNT,1
  2013001.122616816,320,NPNT,1
  2013001.123144816,320,NPNT,1
  2013001.123712816,320,NPNT,1
  2013001.124240816,320,NPNT,1
  2013001.124808816,320,NPNT,1


For the AOPCADMD data notice there are no statistic values.  This is because it is a state
code MSID and so there is no useful meaning for a mean or max.  The final ``raw_vals``
column is the raw telemetered value, while ``vals`` has been translated into the
corresponding state code string.


Details
----------

There are many options controlling ``fetch_ska``, but they can be broken down
into manageable subsets as in the following sections.  This will include detailed
discussion of how to use each of the options.

Desired telemetry
^^^^^^^^^^^^^^^^^^

============== ======================================================
Argument       Description
============== ======================================================
msids          MSID(s) to fetch (string or list of strings)
--start        Start time for data fetch (default=<stop> - 30 days)
--stop         Stop time for data fetch (default=NOW)
--sampling     Data sampling (full | 5min | daily) (default=5min))
--unit_system  Unit system for data (eng | sci | cxc) (default=eng)
============== ======================================================

The first argument ``msids`` is the only one that always has to be provided.  It should be
either a single string like ``COBSRQID`` or a list of strings like ``TEPHIN
TCYLAFT6 TEIO``.  Note that the MSID is case-insensitive so ``tephin`` is fine.

The ``--start`` and ``--stop`` arguments are typically a string like ``2012:001`` or
``2012:001:02:03:04`` (ISO time) or ``2012001.020304`` (GRETA time).  If not provided
then the last 30 days of telemetry will be fetched.

The ``--sampling`` argument will choose between either full-resolution telemetry
or the 5-minute or daily summary statistic values.  The default is ``5min``.

The ``--unit_system`` argument selects the output unit system.  The choices are engineering
units (i.e. what is in the TDB and GRETA), science units (mostly just temperatures in C
instead of F), or CXC units (whatever is in CXC decom, which e.g. has temperatures in K).

Interpolation
^^^^^^^^^^^^^^^

================ ======================================================
Argument         Description
================ ======================================================
--interpolate_dt Interpolate to uniform time steps (secs, default=None)
================ ======================================================

In general different MSIDs will come down in telemetry with different sampling and time
stamps.  Interpolation allows you to put all the MSIDs onto a common time sequence so you
can compare them, plot one against the other, and so forth.  You can see the
`Interpolation`_ section for the gory details, but if you need to have your MSIDs on
a common time sequence then set ``interpolate_dt`` to the desired time step
in seconds.  When interpolating ``ska_fetch`` uses ``filter_bad=True`` and
``union_bad=True`` (as described in `Interpolation`_).

Intervals
^^^^^^^^^^^

================ ======================================================
Argument         Description
================ ======================================================
--remove_events  Remove kadi events expression (default=None)
--select_events  Select kadi events expression (default=None)
================ ======================================================

These arguments allow you to select or remove intervals in the data using the `Kadi event
definitions <http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/kadi/#event-definitions>`_.
For instance we can select times of stable NPM dwells during radiation zones::

  % ska_fetch AOATTER1 AOATTER2 AOATTER3 --start=2014:001 --stop=2014:010 \
              select_events='dwells & rad_zones'

Note the use of a single-quote string for the select events expression.  This makes sure
the expression is treated as a single entity and special characters are not interpreted
by the shell.

The order of processing is to first remove event intervals, then select event intervals.

The expression for ``--remove_events`` or ``--select_events`` can be any logical expression
involving Kadi query names (see the `event definitions table
<http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/kadi/#event-definitions>`_).  The
following string would be valid: ``'dsn_comms | (dwells[pad=-300] & ~eclipses)'``, and for
``select_events`` this would imply selecting telemetry which is either during a DSN pass
or (within a NPM dwell and not during an eclipse).  The ``[pad=-300]`` qualifier means
that a buffer of 300 seconds is applied on each edge to provide padding from the maneuver.
A positive padding expands the event intervals while negative contracts the intervals.

Another example of practical interest is using the LTT bad times event to remove bad times
for long-term trending plots by MSID.  In this case we get daily IRU-2 temps since 2004,
removing known LTT bad times::

  % ska_fetch AIRU2BT --start 2004:001 --sampling=daily --outfile=airu2bt.zip \
                --remove-events='ltt_bads[msid="AIRU2BT"]'

Notice the syntax here which indicates selecting all the LTT bad times corresponding
to ``AIRU2BT``.  See the
`LTT bad times <http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/kadi/#ltt-bad-times>`_
section for more details.

Output
^^^^^^^

================ ======================================================
Argument         Description
================ ======================================================
--time_format    Output time format (secs|date|greta|jd|..., default=secs)
--outfile        Output file name (default='fetch.zip')
================ ======================================================

By default the ``times`` column for each MSID output is provided in the format of seconds
since 1998.0 (CXC seconds).  The ``time_format`` argument allows selecting any time format
supported by `Chandra.Time
<http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/pydocs/Chandra.Time.html>`_.  A common
option for FOT analysis will be ``greta``.

The MSID set will always be written out as a compressed zip archive with the given name
(or ``fetch.zip`` if not provided).  This archive will contain one or more CSV files
corresponding to the MSIDs in the set.

Process control
^^^^^^^^^^^^^^^^^^

================ ======================================================
Argument         Description
================ ======================================================
--quiet          Suppress run-time logging output (default=False)
--max_fetch_Mb   Max allowed memory (Mb) for fetching (default=1000)
--max_output_Mb  Max allowed memory (Mb) for output (default=100)
================ ======================================================

Normally ``ska_fetch`` outputs a few lines of progress information as it is processing the
request.  To disable this logging use the ``--quiet`` flag.

The next two arguments are in place to prevent accidentally doing a huge query that will
consume all available memory or generate a large file that will be slow to read.  For
instance getting all the gyro count data for the mission will take more than 70 Gb of
memory.

The ``--max_fetch_Mb`` argument specifies how much memory the fetched MSID set can
take.  This has a default of 1000 Mb = 1 Gb.

The ``--max_output_Mb`` checks the size of the actual output MSID set (the uncompressed
binary in memory), which may be smaller than the fetch object if data sampling has been
reduced via the ``--interpolate_dt`` argument.  This has a default of 100 Mb.

As an example of what happens if you run into the limits, here is an attempt at the
aforementioned gyro counts query::

  % ska_fetch AOGYRCT1 AOGYRCT2 AOGYRCT3 AOGYRCT4 --start=2000:001 --sampling=full
  Fetching full-resolution data for MSIDS=['AOGYRCT1', 'AOGYRCT2', 'AOGYRCT3', 'AOGYRCT4']
    from 2000:001:12:00:00.000 to 2014:065:17:35:42.347

  ********************************************************************************
  ERROR: Requested fetch requires 76821.73 Mb vs. limit of 1000.00 Mb
  ********************************************************************************

Both of the defaults here are relatively conservative, and with experience you can set
larger values.
