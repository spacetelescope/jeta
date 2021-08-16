import numpy as np
from jeta.archive import fetch

def paginate(msid, draw, idx0=0, length=10):
    """ A simple function to return a page of the full resolution data.
    """
    results = fetch.MSID(msid)

    start = idx0
    stop = idx0 + length

    times = results.times[start:stop]
    values = results.vals[start:stop]

    return {
        'draw': draw,
        'data': [times, values],
        'recordsTotal': len(results),
        'recordsFiltered': len(times)
    }