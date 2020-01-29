The Ska engineering telemetry archive tools support a wide
range of formats for representing date-time stamps.  Note that within this and
other documents for this tool suite, the words 'time' and 'date' are used
interchangably to mean a date-time stamp.

The available formats are listed in the table below:

============ =============================================== =======
 Format      Description                                     System
============ =============================================== =======
  secs       Elapsed seconds since 1998-01-01T00:00:00       tt
  numday     DDDD:hh:mm:ss.ss... Elapsed days and time       utc
  jd*        Julian Day                                      utc
  mjd*       Modified Julian Day = JD - 2400000.5            utc
  date       YYYY:DDD:hh:mm:ss.ss..                          utc
  caldate    YYYYMonDD at hh:mm:ss.ss..                      utc
  fits       FITS date/time format YYYY-MM-DDThh:mm:ss.ss..  tt
  unix*      Unix time (since 1970.0)                        utc
  greta      YYYYDDD.hhmmss[sss]                             utc
============ =============================================== =======

``*`` Ambiguous for input parsing and only available as output formats.

For the ``date`` format one can supply only YYYY:DDD in which case 12:00:00.000
is implied.

The default time "System" for the different formats is either ``tt``
(Terrestrial Time) or ``utc`` (UTC).  Since TT differs from UTC by around 64
seconds it is important to be consistent in specifying the time format.

