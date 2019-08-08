The Ska engineering telemetry archive is a suite of tools and data products
that make available the majority of all Chandra engineering telemetry since the
start of year 2000.  This includes about 6300 MSIDs.  The telemetry are stored
in a way that allows for very fast and efficient retrieval into memory.
Typical retrieve rates are around 10^7 samples/sec.  For an MSID sampled once
per second this translates to about 3 sec per year of data.

The engineering telemetry archive consists of:

* Tools to ingest and compress telemetry from the CXC Chandra archive products.
* Compressed telemetry files in HDF5 format.  Each MSID has three associated products:

  - Full time-resolution data: time, value, quality
  - 5-minute statistics: min, max, mean, sampled value, number of samples
  - Daily statistics: min, max, mean, sampled value, standard deviation, percentiles (1,
    5, 16, 50, 84, 95, 99), number of samples.
* A python module to retrieve telemetry values.

running jeta with celery:

redis need to run first

# The worker to run tasks
celery -A jeta.ingest.controller worker --loglevel=info

# The default monitoring interface for tasks
celery -A jeta.ingest.controller flower --loglevel=info

#