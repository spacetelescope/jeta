Fixing bad values
==================

Excluding bad values
--------------------

Bad values may be reported by users or show up as a warning in Eng archive processing, e.g.::

  Errors in files:
  /proj/sot/ska/data/eng_archive/logs/eng_archive.log
  ** ERROR - line 27073: /proj/sot/ska/share/eng_archive/update_archive.py:382: RuntimeWarning: overflow encountered in square

From the log identify the MSID for the bad value(s).  Fetch recent data and make a plot to confirm an
obvious issue.  For instance::

  >>> dat = fetch.MSID('aomantim', '2013:230', '2013:235')
  >>> dat.plot()  # shows a huge spike around day 233
  >>> bad = dat.vals > 100000
  >>> DateTime(dat.times[bad]).date
  array(['2013:233:12:19:26.158'],
        dtype='|S21')

This is a single bad point and can be fixed by providing that precise date
as the --start argument.  To do a dry run (which is the default) to see
what the tool will do::

  % ska
  % cd ~/git/eng_archive
  % ./fix_bad_values.py --msid aomantim --start=2013:233:12:19:26.158 \\
       --data-root=/proj/sot/ska/data/eng_archive

  ** If something gets corrupted then there is the NetApp snapshot for recovery **

  ** DRY RUN **

  Fixing MSID AOMANTIM h5 file
  Reading TIME file /proj/sot/ska/data/eng_archive/data/pcad8eng/TIME.h5
  Reading msid file /proj/sot/ska/data/eng_archive/data/pcad8eng/AOMANTIM.h5
  AOMANTIM.data[52489035] = 2.57904159479e+25
  Changing AOMANTIM.quality[52489035] from False to True
  ...

This also outputs updates for the 5min and daily stats, but in the dry run it will still
produce bad values because the actual full-sample values will not have been
updated so the stats are still computed with the bad value(s).

One you are satisfied with the above output then run for real using the --run flag.
Make sure the statistics look good.  You'll need to do the same on the GRETA network
in ska-test.

**IMPORTANT!**

Copy the command you used in the "Record of Fixes" log below, as this is the only
permanent record of these hand-fixes.  Then commit this updated file to git and push to
origin.

Setting new values
------------------

In cases such as safe mode where there is incorrect telemetry over a long time range, it
can be helpful to substitute new "correct" values.  The poster-child here is DP_PITCH
which gets reported in CXC telemetry incorrectly.  The fix procedure here is similar
to above except that the ``--value=<value>`` command line argument is supplied::

  % ska
  % cd ~/git/eng_archive
  % ./fix_bad_values.py --msid dp_pitch --value=90.0 --start=2011:188 --stop=2011:190 \\
       --data-root=/proj/sot/ska/data/eng_archive [--run]

Again it is crucial to record each of these commands and validate the results
by plotting and printing data values after applying the fix.

Record of fixes
---------------
::

  ./fix_bad_values.py --msid aorate3 --start=2013:146:16:12:44.600 --data-root=/proj/sot/ska/data/eng_archive --run
  ./fix_bad_values.py --msid aorate1 --start=2007:310:22:10:02.951 --data-root=/proj/sot/ska/data/eng_archive --run
  ./fix_bad_values.py --msid aomantim --start=2012:324:16:41:15.873 --data-root=/proj/sot/ska/data/eng_archive --run
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid OHRTHR55
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_EE_AXIAL
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_EE_BULK
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_EE_THERM
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HAAG
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HMAX35
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HMIN35
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HMCSAVE
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HRMA_AVE
  ./fix_bad_values.py --start '2013:197:03:50:12.387' --stop '2013:197:03:50:25.016' --data-root /proj/sot/ska/data/eng_archive --run --msid DP_HRMHCHK
  ./fix_bad_values.py --msid aomantim --start=2013:233:12:19:26.158 --data-root=/proj/sot/ska/data/eng_archive --run

  # Safe mode dp_pitch
  ./fix_bad_values.py --start '1999:229:20:17:50.616' --stop '1999:231:01:29:21.816' --data-root /proj/sot/ska/data/eng_archive --msid DP_PITCH --value=90.0 --run
  ./fix_bad_values.py --start '1999:269:20:22:50.616' --stop '1999:270:08:22:15.416' --data-root /proj/sot/ska/data/eng_archive --msid DP_PITCH --value=90.0 --run
  ./fix_bad_values.py --start '2000:048:08:08:54.216' --stop '2000:049:16:48:40.000' --data-root /proj/sot/ska/data/eng_archive --msid DP_PITCH --value=90.0 --run
  ./fix_bad_values.py --start '2011:187:12:28:53.816' --stop '2011:192:03:54:20.000' --data-root /proj/sot/ska/data/eng_archive --msid DP_PITCH --value=90.0 --run
  ./fix_bad_values.py --start '2012:150:03:33:09.816' --stop '2012:152:02:01:10.000' --data-root /proj/sot/ska/data/eng_archive --msid DP_PITCH --value=90.0 --run


Appendix: plot safe mode pitch values
-------------------------------------
The post-2000 safe mode end times for fixing DP_PITCH were determined
with the following script and by zooming into the time when PITCH
returns to about 90 degrees (indicating on-board attitude is correct).
::

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
