import numpy as np
from jeta.archive import fetch

def paginate_stat(msid, draw, idx0=0, idx1=None, interval='5min'):
    """ A simple function to return a page of the full resolution data.
    """

    results = fetch.MSID(msid, interval=interval)

    times = results.times[idx0:idx0 + idx1]
    values = results.vals[idx0:idx0 + idx1]

    return {
        'draw': draw,
        'data': [times, values],
        'recordsTotal': len(results),
        'recordsFiltered': len(times)
    }


def paginate(msid, draw, idx0=0, idx1=None):
    """ A simple function to return a page of the full resolution data.
    """

    results = fetch.MSID(msid)

    times = results.times[idx0:idx0 + idx1]
    values = results.vals[idx0:idx0 + idx1]

    return {
        'draw': draw,
        'data': [times, values],
        'recordsTotal': len(results),
        'recordsFiltered': len(times)
    }