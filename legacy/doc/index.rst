==================================
 Ska Engineering Telemetry Archive
==================================
.. include:: references.rst

Overview
--------

The Ska engineering telemetry archive is a suite of tools and data products
that make available the majority of all Chandra engineering telemetry since the
start of the mission (1999:204).  This includes about 6300 MSIDs.  The telemetry are stored
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

Pseudo-MSIDs
----------------------------

A small selection of pseudo-MSIDs that do not come in the engineering telemetry
stream are also available in the archive.  These are:

* SIM telemetry: SIM position and moving status (deprecated)
* EPHIN telemetry: level-0 science telemetry from EPHIN
* ACIS DEA housekeeping: status from the DEA (including detector focal plane temperature)
* Ephemeris: predictive and definitive orbital (Chandra), solar, and lunar ephemeris values
* Derived parameters: values computed from other MSIDs in the archive (ACIS power, orbital elements, PCAD, thermal)

For details see the documentation on Pseudo-MSIDs in the engineering archive.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   tutorial
   fetch_tutorial
   ska_fetch
   pseudo_msids

API docs
--------

.. toctree::
   :maxdepth: 1

   fetch
   plot
   utils

.. toctree::
   :hidden:

   date_time_formats
   fetch_tutorial_standalone
   ipython_tutorial
   matplotlib_tutorial
   numpy_tutorial
   references
