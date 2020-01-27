# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make a unit_system consistent with usual CXC Science units where all temperatures
are in degC instead of Kelvins.  Otherwise leave the CXC units untouched.
"""
import cPickle as pickle

units_cxc = pickle.load(open('units_cxc.pkl'))
units_sci = dict((msid, 'DEGC') for msid, unit in units_cxc.items()
                 if unit == 'K' or unit == 'deltaK')
pickle.dump(units_sci, open('units_sci.pkl', 'wb'))
