# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from ..utils import get_fetch_size
from .. import fetch


def test_get_fetch_size_functionality():
    """
    Various functionality tests for estimating fetch memory usage.
    """
    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2011:001')
    assert fetch_mb, out_mb == (399.97, 399.97)

    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2011:001', interpolate_dt=1.025 * 10)
    assert fetch_mb, out_mb == (399.97, 40.0)

    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2011:001', fast=False, stat='5min')
    assert fetch_mb, out_mb == (1.92, 1.92)

    # 5min stat
    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2011:001', stat='5min')
    assert fetch_mb, out_mb == (-1, -1)

    # Too short
    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2010:030')
    assert fetch_mb, out_mb == (-1, -1)


def test_get_fetch_size_accuracy():
    """
    Does it give the right answer?
    """
    # By hand for stat
    dat = fetch.MSID('aopcadmd', '2010:001', '2011:001', stat='5min')
    fetch_bytes = sum(getattr(dat, attr).nbytes for attr in dat.colnames)

    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2010:001', '2011:001', stat='5min',
                                      interpolate_dt=328 * 2, fast=False)
    assert np.isclose(fetch_mb, fetch_bytes / 1e6, rtol=0.0, atol=0.01)

    # Now interpolate to 10 minute intervals
    dat.interpolate(328.0 * 2)
    fetch_bytes = sum(getattr(dat, attr).nbytes for attr in dat.colnames)
    assert np.isclose(out_mb, fetch_bytes / 1e6, rtol=0.0, atol=0.01)

    # By hand for full resolution
    dat = fetch.MSID('aopcadmd', '2011:001', '2011:010')
    fetch_bytes = sum(getattr(dat, attr).nbytes for attr in dat.colnames)

    fetch_mb, out_mb = get_fetch_size('aopcadmd', '2011:001', '2011:010',
                                      interpolate_dt=328 * 2, fast=False)
    assert np.isclose(fetch_mb, fetch_bytes / 1e6, rtol=0.0, atol=0.01)

    # Now interpolate to 10 minute intervals
    dat.interpolate(328.0 * 2)
    fetch_bytes = sum(getattr(dat, attr).nbytes for attr in dat.colnames)
    assert np.isclose(out_mb, fetch_bytes / 1e6, rtol=0.0, atol=0.01)
