# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make a unit_system consistent with usual OCC/FOT engineering units via P009.
"""

import pickle
from copy import copy

import Ska.engarchive.converters
import Ska.tdb

units_cxc = pickle.load(open('units_cxc.pkl', 'rb'))
units_eng = copy(units_cxc)

# Update with units from TMSRMENT table of the TDB
dat = Ska.tdb.tables['tmsrment']
tdb_units = {str(msid.upper()): str(unit)  # convert from numpy type to pure Python
             for msid, unit in zip(dat['MSID'], dat['ENG_UNIT'])}
units_eng.update({msid: tdb_units[msid] for msid, unit in units_eng.items()
                  if msid in tdb_units})

# Any other MSIDs that still have units of 'K' are converted to DEGC
units_eng.update({msid: 'DEGC' for msid, unit in units_eng.items()
                  if unit in ['K', 'deltaK']})

# Use info about DEA HK telemetry from converters to add units
for col in Ska.engarchive.converters._get_deahk_cols():
    if 'unit' in col and col['unit'] == 'K':
        units_eng[col['name'].upper()] = 'DEGC'

# Make STEP unique for SIM MSIDs
units_eng['3LDRTPOS'] = 'TSCSTEP'
units_eng['3TSCPOS'] = 'TSCSTEP'
units_eng['3FAPOS'] = 'FASTEP'
units_eng['3MRMMXMV'] = 'PWMSTEP'

units_eng['HKEBOXTEMP'] = 'DEGF'

pickle.dump(units_eng, open('units_eng.pkl', 'wb'))
