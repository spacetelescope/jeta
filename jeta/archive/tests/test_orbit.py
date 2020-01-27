# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from ..derived import orbit


def test_orbital_elements():
    """
    Test orbital elements calculation vs. reference
    http://www.castor2.ca/05_OD/01_Gauss/14_Kepler/index.html

    The example there suffers from some roundoff error by using the printed values instead
    of full precision.
    """
    # State vector in m, m/s
    x = 5052.4587e3
    y = 1056.2713e3
    z = 5011.6366e3
    vx = 3.8589872e3
    vy = 4.2763114e3
    vz = -4.8070493e3

    out = orbit.calc_orbital_elements(x, y, z, vx, vy, vz)

    expected = {'semi_major_axis': 7310.8163e3,
                'orbit_period': 103.68323 * 60,
                'eccentricity': 0.015985887,
                'inclination': 71.048202,
                'ascending_node': 211.28377,
                'argument_perigee': 137.7561049,  # 137.75619 in reference (roundoff in example)
                'mean_anomaly': 354.971325}  # 354.9724 in reference (roundoff in example)

    for key in expected:
        assert np.allclose(out[key], expected[key], atol=0.0, rtol=1e-6)
