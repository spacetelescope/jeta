**Pylab**

For interactive data analysis IPython has a special ``--pylab`` command line
option which automatically imports elements of the Numpy and the Matplotlib 
environments.  This provides a Matlab-like environment allowing very simple
and direct commands like::

  % ipython --pylab
  
  x = arange(0, 10, 0.2)
  y = sin(x)
  print x
  plot(x, y)

**Keyboard navigation and history**

One of the most useful features of IPython is the ability to edit and navigate 
you command line history.  This lets you quickly re-do commands, perhaps with a
slight variation based on seeing the last result.  Try cut-n-pasting the above
lines in an IPython session.  This should bring up a plot of a sine wave.  

Now hit up-arrow once and get back the ``plot(x, y)`` line.  Hit the left-arrow
key (not backspace) once and type ``**2`` so that the line reads ``plot(x,
y**2)``.  Now you can hit Return to see the new curve overlayed within the same
plot window.  It is not necessary to forward-space to the end of the line, you
can hit Return with the cursor anywhere in the line.

Now say you want to change the ``x`` values slightly.  One option is to just hit the
up-arrow 5 times, but a much faster way is to remember that the line started
with ``x``, so type ``x`` and then start hitting up-arrow.  Only lines that
start with ``x`` will be displayed and you are immediately at the 
``x = arange(0, 10, 0.2)`` line.  Now use the right-arrow and backspace to change ``10`` to
``15`` and hit Return.  Of course ``y`` needs to be recalculated, so hit ``y``
then up-arrow, then Return.  Finally ``pl`` up-arrow and Return.  Nice and fast!

Bonus points: to speed up by another factor of several, use Ctrl-p (prev) instead of
up-arrow, Ctrl-n (next) instead of down-arrow, Ctrl-f (forward) instead of
right-arrow and Ctrl-b (back) instead of left-arrow.  That way your fingers
never leave the keyboard home keys.  Ctrl-a gets you to the beginning of the
line and Ctrl-e gets you to the end of the line.  Triple bonus: on a Mac or
Windows machine re-map the Caps-lock key to be Control so it's right next to
your left pinky.  How often do you need Caps-lock?

Your command history is saved between sessions (assuming that you exit IPython
gracefully) so that when you start a new IPython you can use up-arrow to re-do
old commands.  You can view your history within the current session by entering
``history``.

**Linux and shell commands**

A select set of useful linux commands are available from the IPython prompt.
These include ``ls`` (list directory), ``pwd`` (print working directory),
``cd`` (change directory), and ``rm`` (remove file).  Any shell command
can be executed by preceding it with an exclamation point "!".

**Tab completion**

IPython has a very useful tab completion feature that can be used both to
complete file names and to inspect python objects.  As a first example do::

  ls /proj/sot/ska/<TAB>

This will list everything that matches ``/proj/sot/ska``.  You can continue
this way searching through files or hit Return to complete the command.

Tab completion also works to inspect object attributes.  Every variable or
constant in python is actually a object with a type and associated attributes
and methods.  For instance try to create a list of 3 numbers::

  a = [3, 1, 2, 4]
  print a
  a.<TAB>

This will show the available methods for ``a``::

  In [17]: a.<TAB>
  a.append   a.extend   a.insert   a.remove   a.sort     
  a.count    a.index    a.pop      a.reverse  

Here you see useful looking functions like ``append`` or ``sort`` which you can
get help for and use::

  help a.sort
  a.sort()
  print a

For a more concrete example, say you want to fetch some daily telemetry values
but forgot exactly how to do the query and what are the available columns.  Use
help and TAB completion to remind yourself::

  import Ska.engarchive.fetch as fetch
  help fetch  
  tephin = fetch.MSID('tephin', '2009:001', '2009:002', stat='daily')
  tephin.<TAB>
   tephin.bads               tephin.midvals            tephin.samples
   tephin.colnames           tephin.mins               tephin.stat
   tephin.content            tephin.msid               tephin.state_codes
   tephin.datestart          tephin.MSID               tephin.state_intervals
   tephin.datestop           tephin.p01s               tephin.stds
   tephin.dt                 tephin.p05s               tephin.tdb
   tephin.filter_bad         tephin.p16s               tephin.times
   tephin.filter_bad_times   tephin.p50s               tephin.tstart
   tephin.indexes            tephin.p84s               tephin.tstop
   tephin.iplot              tephin.p95s               tephin.unit
   tephin.logical_intervals  tephin.p99s               tephin.vals
   tephin.maxes              tephin.plot               tephin.write_zip
   tephin.means              tephin.raw_vals           

OK, now you remember you wanted ``times`` and ``maxes``.  But look, there are
other tidbits there for free that look interesting.  So go ahead and print a few::

  print tephin.colnames
  print tephin.dt
  print tephin.MSID

To make it even easier you don't usually have to use ``print``.  Try the
following::

  tephin.colnames
  tephin.dt
  tephin.MSID

Don't be scared to try printing an array value (e.g. ``tephin.vals``) even if
it is a billion elements long.  Numpy will only print an abbreviated version if
it is too long.  But beware that this applies to Numpy arrays which as we'll
see are a special version of generic python lists.  If you print a
billion-element python list you'll be waiting for a while.
