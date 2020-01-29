# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Create the default unit system as found in the CXC telemetry FITS files.
"""
from __future__ import print_function

import os
import re
import glob
import argparse
import cPickle as pickle
import pyfits
import pyyaks

from Ska.engarchive.converters import _get_deahk_cols, CXC_TO_MSID
from Ska.engarchive import file_defs
from Ska.engarchive import fetch


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Dry run (no actual file or database updates)")
    return parser.parse_args(args)

opt = get_options()

ft = fetch.ft
arch_files = pyyaks.context.ContextDict('update_archive.arch_files',
                                        basedir=os.path.join(file_defs.SKA, 'data', 'eng_archive'))
arch_files.update(file_defs.arch_files)

units = {}

# CXC content types
contents = set(fetch.content.values())

for content in sorted(contents):
    if content.startswith('dp_'):
        continue
    ft['content'] = content

    dir_ = arch_files['archrootdir'].abs
    print('Finding units in', dir_)

    # Get the most recent directory
    years = sorted([yr for yr in os.listdir(dir_) if re.match(r'\d{4}', yr)])
    dir_ = os.path.join(dir_, years[-1])

    days = sorted([day for day in os.listdir(dir_) if re.match(r'\d{3}', day)])
    dir_ = os.path.join(dir_, days[-1])

    files = glob.glob(os.path.join(dir_, '*.fits.gz'))
    if not files:
        print('No {} fits files in {}'.format(content, dir_))
        continue

    print('Reading', files[0])
    hdus = pyfits.open(os.path.join(dir_, files[0]))
    cols = hdus[1].columns
    for msid, unit in zip(cols.names, cols.units):
        unit = unit.strip()
        if unit:
            msid = msid.upper()
            if content in CXC_TO_MSID:
                msid = CXC_TO_MSID[content].get(msid, msid)
            if re.match(r'(orbit|lunar|solar|angle)ephem', content):
                msid = '{}_{}'.format(content.upper(), msid)
            units[msid.upper()] = unit
    hdus.close()

# AFAIK these are the only temperature MSIDs that are actually temperature
# differences and which require special handling on conversion.
relative_temp_msids = (
    'OHRMGRD3',  # RT 500 TO RT 502: CAP GRADIENT MONITOR
    'OHRMGRD6',  # RT 503 TO RT 501: CAP GRADIENT MONITOR
    'OOBAGRD3',  # RT 505 TO RT 504: PERISCOPE GRADIENT MONITOR
    'OOBAGRD6',  # RT 507 TO RT 506: PERISCOPE GRADIENT MONITOR
    )

for msid in relative_temp_msids:
    units[msid] = 'deltaK'

units['3MRMMXMV'] = 'PWM'

# Use info about DEA HK telemetry from converters to add units
for col in _get_deahk_cols():
    if 'unit' in col:
        units[col['name'].upper()] = col['unit']

if not opt.dry_run:
    pickle.dump(units, open('units_cxc.pkl', 'wb'))
else:
    print(repr(units))
