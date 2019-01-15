# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Compatibility with ancient plotting scripts.
"""

from .fetch import MSID


def fetch_arrays(start, stop, msids):
    """
    Fetch data for ``msids`` from the telemetry archive as arrays.

    This routine is deprecated and is retained only for back-compatibility with
    old plotting analysis scripts.

    The telemetry values are returned in three dictionaries: ``times``,
    ``values``, and ``quals``.  Each of these dictionaries contains key-value
    pairs for each of the input ``msids``.

    :param start: start date of telemetry (Chandra.Time compatible)
    :param stop: stop date of telemetry
    :param msids: list of MSIDs (case-insensitive)

    :returns: times, values, quals
    """
    times = {}
    values = {}
    quals = {}

    for msid in msids:
        m = MSID(msid, start, stop)
        times[msid] = m.times
        values[msid] = m.vals
        quals[msid] = m.bads

    return times, values, quals


def fetch_array(start, stop, msid):
    """
    Fetch data for single ``msid`` from the telemetry archive as an array.

    This routine is deprecated and is retained only for back-compatibility with
    old plotting analysis scripts.

    The telemetry values are returned in three arrays: ``times``, ``values``,
    and ``quals``.

    :param start: start date of telemetry (Chandra.Time compatible)
    :param stop: stop date of telemetry
    :param msid: MSID (case-insensitive)

    :returns: times, values, quals
    """

    m = MSID(msid, start, stop)
    times = m.times
    values = m.vals
    quals = m.bads

    return times, values, quals
