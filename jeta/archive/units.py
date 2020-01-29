# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Basic units handling and conversion.

>>> eng = units.units['eng']
>>> cxc = units.units['cxc']
>>> comb = set()
>>> for e in eng.keys():
...    comb.add((eng[e], cxc[e]))

>>> comb   # ** entries require explicit conversion
{('AMP', 'A'),
 ('ASEC', 'arcsec'),
 ('DEG', 'deg'),
 ('DEGC', 'K'),        **
 ('DEGF', 'K'),        **
 ('DEGF', 'deltaK'),   **
 ('DEGPS', 'deg/s'),
 ('FASTEP', 'mm'),     **
 ('FTLB', 'J'),        **
 ('FTLBSEC', 'J*s'),   **
 ('KHZ', 'kHz'),
 ('KM', 'km'),
 ('KMPS', 'km/s'),
 ('MAMP', 'mA'),
 ('MIN', 's'),         **
 ('MSEC', 'ms'),
 ('PSIA', 'kPa'),      **
 ('PWMSTEP', 'PWM'),   **
 ('RAD', 'rad'),
 ('RADPS', 'rad/s'),
 ('RADSS', 'rad/s**2'),
 ('SEC', 's'),
 ('TORR', 'kPa'),      **
 ('TSCSTEP', 'mm'),    **
 ('V', 'V'),
 ('VDC', 'V'),
 ('W', 'W')}
"""
from __future__ import print_function, division, absolute_import

import os
from six.moves import cPickle as pickle
import logging
import warnings

import numpy as np


class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logger = logging.getLogger('Ska.engarchive.units')
logger.addHandler(NullHandler())
logger.propagate = False

# This is effectively a singleton class
SYSTEMS = set(('cxc', 'eng', 'sci'))
module_dir = os.environ['JETA_SCRIPTS']  # os.path.dirname(__file__)

units = {}
units['system'] = 'cxc'
units['cxc'] = pickle.load(open(os.path.join(module_dir, 'units_cxc.pkl'), 'rb'))


# Equivalent unit descriptors used in 'eng' and 'cxc' units
equiv_units = set([('AMP', 'A'),
                   ('ASEC', 'arcsec'),
                   ('DEG', 'deg'),
                   ('DEGPS', 'deg/s'),
                   ('KHZ', 'kHz'),
                   ('KM', 'km'),
                   ('KMPS', 'km/s'),
                   ('MAMP', 'mA'),
                   ('MSEC', 'ms'),
                   ('RAD', 'rad'),
                   ('RADPS', 'rad/s'),
                   ('RADSS', 'rad/s**2'),
                   ('SEC', 's'),
                   ('V', 'V'),
                   ('VDC', 'V'),
                   ('W', 'W'),
                   (None, None)])
equiv_units.update((u2, u1) for u1, u2 in list(equiv_units))


def F_to_C(vals, delta_val=False):
    if delta_val:
        return vals / 1.8
    else:
        return (vals - 32.0) / 1.8


def C_to_K(vals, delta_val=False):
    if delta_val:
        return vals
    else:
        return vals + 273.15


def K_to_C(vals, delta_val=False):
    if delta_val:
        return vals
    else:
        return vals - 273.15


def K_to_F(vals, delta_val=False):
    if delta_val:
        return vals * 1.8
    else:
        return vals * 1.8 - 459.67  # (vals - 273.15) * 1.8 + 32


def F_to_K(vals, delta_val=False):
    if delta_val:
        return vals / 1.8
    else:
        return (vals + 459.67) / 1.8


def FASTEP_to_mm(vals, delta_val=False):
    """
    Use CXC calibration value to convert from focus assembly steps to mm.
    """
    fastep = (1.47906994e-3 * vals + 3.5723322e-8 * vals**2 + -1.08492544e-12 * vals**3 +
              3.9803832e-17 * vals**4 + 5.29336e-21 * vals**5 + 1.020064e-25 * vals**6)
    return fastep


def mm_to_FASTEP(vals, delta_val=False):
    """
    # compute mm from simulated integral step values and invert the CXC calibration
    # given below:
    x = np.arange(-5000.0, 0.0)  # step
    y = (1.47906994e-3  *   x +  3.5723322e-8 *   x**2 +  -1.08492544e-12  *   x**3 +
         3.9803832e-17  *   x**4 +  5.29336e-21  *  x**5 +  1.020064e-25  *   x**6)
    r = np.polyfit(y, x, 8)
    """
    r = np.array([-1.26507734e-05, -2.02499464e-04, -1.86504522e-03,
                  -5.25689124e-03, -5.75639912e-02, 5.60935786e-01,
                  -1.10595209e+01, 6.76094720e+02, -4.34121454e-04])
    x_step = np.round(np.polyval(r, vals), decimals=2)
    return x_step


def mult(scale_factor, decimals=None):
    def convert(vals, delta_val=False):
        result = vals * scale_factor
        if decimals is not None:
            result = np.round(result, decimals=decimals)
        return result
    return convert


def divide(scale_factor, decimals=None):
    return mult(1.0 / scale_factor, decimals)


converters = {
    # CXC units to Eng or Sci
    ('J', 'FTLB'): mult(0.7376),
    ('J*s', 'FTLBSEC'): mult(0.7376),
    ('K', 'DEGC'): K_to_C,
    ('K', 'DEGF'): K_to_F,
    ('deltaK', 'DEGF'): mult(1.8),
    ('kPa', 'PSIA'): mult(0.145),
    ('kPa', 'TORR'): mult(7.501),
    ('mm', 'TSCSTEP'): mult(1.0 / 0.00251431530156, decimals=3),
    ('mm', 'FASTEP'): mm_to_FASTEP,
    ('PWM', 'PWMSTEP'): mult(16),

    # Eng units to CXC or Sci
    ('DEGC', 'K'): C_to_K,
    ('DEGF', 'K'): F_to_K,
    ('DEGF', 'deltaK'): divide(1.8),
    ('DEGF', 'DEGC'): F_to_C,
    ('FASTEP', 'mm'): FASTEP_to_mm,
    ('FTLB', 'J'): divide(0.7376),
    ('FTLBSEC', 'J*s'): divide(0.7376),
    ('MIN', 's'): mult(60),
    ('PSIA', 'kPa'): divide(0.145),
    ('PWMSTEP', 'PWM'): divide(16),
    ('TORR', 'kPa'): divide(7.501),
    ('TSCSTEP', 'mm'): mult(0.00251431530156),
    }


def load_units(unit_system):
    """Load units definitions for unit_system if not already loaded.
    """
    if unit_system not in SYSTEMS:
        raise ValueError('unit_system must be in {}'.format(SYSTEMS))

    if unit_system not in units:
        filename = os.path.join(module_dir,
                                'units_{0}.pkl'.format(unit_system))
        units[unit_system] = pickle.load(open(filename, 'rb'))


def set_units(unit_system):
    """Set conversion unit system.  The input ``unit_system`` must be a string.
    """
    load_units(unit_system)
    units['system'] = unit_system


def get_msid_unit(msid):
    MSID = msid.upper()
    return units[units['system']].get(MSID)


def convert(msid, vals, delta_val=False):
    MSID = msid.upper()
    conversion = (units['cxc'].get(MSID), get_msid_unit(MSID))
    try:
        vals = converters[conversion](vals, delta_val)
    except KeyError:
        pass
    return vals


class Units(dict):
    """
    Provide access to units via object-oriented replacement for
    Ska.engarchive.units module.

    This class is weird because it was designed to conform to the
    existing API where units.units was a module dict that had keys
    'system', 'cxc', and possibly others.  But this implementation now
    allows for non-interacting units systems in fetch, fetch_eng, and
    fetch_sci.

    It only allows a single key "system" to be set
    """

    def __init__(self, system='cxc'):
        super(Units, self).__init__(system=system)

    def __getitem__(self, item):
        if item in SYSTEMS:
            load_units(item)
            return units[item]
        else:
            return dict.__getitem__(self, item)

    def __setitem__(self, item, val):
        if item == 'system':
            load_units(val)
            dict.__setitem__(self, item, val)
        else:
            raise KeyError('In Units object only "system" key is settable')

    def set_units(self, unit_system):
        """Set conversion unit system.  The input ``unit_system`` must be a
        string.
        """
        self['system'] = unit_system

    def get_msid_unit(self, msid):
        MSID = msid.upper()
        system = self['system']
        cxc_unit = self['cxc'].get(MSID)
        system_unit = self[system].get(MSID, cxc_unit)  # WHY this default of cxc_unit??
        return system_unit

    def convert(self, msid, vals, delta_val=False, from_system='cxc'):
        MSID = msid.upper()
        conversion = (self[from_system].get(MSID), self.get_msid_unit(MSID))

        if conversion[0] == conversion[1] or conversion in equiv_units:
            return vals

        if conversion not in converters:
            warnings.warn('\n\n\n**** WARNING ****\n'
                          'For MSID {} the requested unit conversion from {} to {}\n'
                          'does not have a defined transformation function.\n'
                          'You may be getting incorrect results now.\n\n'
                          'PLEASE REPORT THIS to the Ska developers!\n'
                          .format(MSID, conversion[0], conversion[1]))

        vals = converters[conversion](vals, delta_val)

        return vals
