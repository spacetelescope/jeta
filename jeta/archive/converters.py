# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division, absolute_import

import logging
import numpy
import sys
from collections import OrderedDict

from six.moves import zip

import numpy as np
import Ska.Numpy
from Chandra.Time import DateTime
import Ska.tdb

from . import units

MODULE = sys.modules[__name__]
logger = logging.getLogger('engarchive')


class NoValidDataError(Exception):
    pass


class DataShapeError(Exception):
    pass


def quality_index(dat, colname):
    """Return the index for `colname` in `dat`"""
    colname = colname.split(':')[0]
    return list(dat.dtype.names).index(colname)


def numpy_converter(dat):
    return Ska.Numpy.structured_array(dat) #Ska.Numpy.structured_array(dat, colnames=dat.dtype.names)


def convert(dat, content):
    # Zero-length file results in `dat is None`
    if dat is None:
        raise NoValidDataError

    try:
        converter = getattr(MODULE, content.lower())
    except AttributeError:
        converter = numpy_converter

    return converter(dat)


def generic_converter(prefix=None, add_quality=False, aliases=None):
    """Convert an input FITS recarray assuming that it has a TIME column.
    If ``add_prefix`` is set then add ``content_`` as a prefix
    to the data column names.  If ``add_quality`` is set then add a QUALITY
    column with all values False.
    """
    def _convert(dat):
        colnames = dat.dtype.names
        colnames_out = [x.upper() for x in colnames]
        if aliases:
            colnames_out = [aliases.get(x, x).upper() for x in colnames_out]
        if prefix:
            # Note to self: never change an enclosed reference, i.e. don't do
            # prefix = prefix.upper() + '_'
            # You will lose an hour again figuring this out if so.
            PREFIX = prefix.upper() + '_'
            colnames_out = [(x if x in ('TIME', 'QUALITY') else PREFIX + x)
                            for x in colnames_out]

        arrays = [dat.field(x) for x in colnames]

        if add_quality:
            descrs = [(x,) + y[1:] for x, y in zip(colnames_out, dat.dtype.descr)]
            quals = numpy.zeros((len(dat), len(colnames) + 1), dtype=numpy.bool)
            descrs += [('QUALITY', numpy.bool, (len(colnames) + 1,))]
            arrays += [quals]
        else:
            descrs = [(name, array.dtype.str, array.shape[1:])
                      for name, array in zip(colnames_out, arrays)]

        return numpy.rec.fromarrays(arrays, dtype=descrs)

    return _convert


def get_bit_array(dat, in_name, out_name, bit_index):
    bit_indexes = [int(bi) for bi in bit_index.split(',')]
    bit_index = max(bit_indexes)

    if dat[in_name].shape[1] < bit_index:
        raise DataShapeError('column {} has shape {} but need at least {}'
                             .format(in_name, dat[in_name].shape[1], bit_index + 1))

    if len(bit_indexes) > 1:
        mult = 1
        out_array = np.zeros(len(dat), dtype=np.uint32)  # no more than 32 bit indexes
        for bit_index in reversed(bit_indexes):
            out_array += np.where(dat[in_name][:, bit_index], mult, 0)
            mult *= 2
    else:
        try:
            tscs = Ska.tdb.msids[out_name].Tsc
            scs = {tsc['LOW_RAW_COUNT']: tsc['STATE_CODE'] for tsc in tscs}
        except (KeyError, AttributeError):
            scs = ['OFF', 'ON ']

        # CXC telemetry stores state code vals with trailing spaces so all match
        # in length.  Annoying, but reproduce this here for consistency so
        # fetch Msid.raw_vals does the right thing.
        max_len = max(len(sc) for sc in scs.values())
        fmtstr = '{:' + str(max_len) + 's}'
        scs = [fmtstr.format(val) for key, val in scs.items()]

        out_array = np.where(dat[in_name][:, bit_index], scs[1], scs[0])

    return out_array


def generic_converter2(msid_cxc_map, default_dtypes=None):
    """Convert an input FITS recarray assuming that it has a TIME column.  Use the
    ``msid_cxc_map`` to define the list of output eng archive MSIDs (keys) and the
    corresponding colnames in the CXC archive FITS file (values).

    The CXC values can contain an optional bit specifier in the form <colname>:<N>
    where N is the bit selector referenced from 0 as the leftmost bit.

    :param msid_cxc_map: dict of out_name => in_name mapping
    """
    def _convert(dat):
        # Make quality bool array with entries for TIME, QUALITY, then all other cols
        out_names = ['TIME', 'QUALITY'] + list(msid_cxc_map.keys())
        out_quality = np.zeros(shape=(len(dat), len(out_names)), dtype=np.bool)
        out_arrays = {'TIME': dat['TIME'],
                      'QUALITY': out_quality}

        for out_name, in_name in msid_cxc_map.items():
            if ':' in in_name:
                in_name, bit_index = in_name.split(':')
                out_array = get_bit_array(dat, in_name, out_name, bit_index)
                quality = dat['QUALITY'][:, quality_index(dat, in_name)]
            else:
                if in_name in dat.dtype.names:
                    out_array = dat[in_name]
                    quality = dat['QUALITY'][:, quality_index(dat, in_name)]
                else:
                    # Handle column that is intermittently available in `dat` by using the
                    # supplied default dtype.  Quality is True (missing) everywhere.
                    out_array = np.zeros(shape=len(dat), dtype=default_dtypes[out_name])
                    quality = True

            assert out_array.ndim == 1
            out_arrays[out_name] = out_array
            out_quality[:, out_names.index(out_name)] = quality

        out = Ska.Numpy.structured_array(out_arrays, out_names)
        return out

    return _convert

orbitephem0 = generic_converter('orbitephem0', add_quality=True)
lunarephem0 = generic_converter('lunarephem0', add_quality=True)
solarephem0 = generic_converter('solarephem0', add_quality=True)
orbitephem1 = generic_converter('orbitephem1', add_quality=True)
lunarephem1 = generic_converter('lunarephem1', add_quality=True)
solarephem1 = generic_converter('solarephem1', add_quality=True)
angleephem = generic_converter(add_quality=True)


def parse_alias_str(alias_str, invert=False):
    aliases = OrderedDict()
    for line in alias_str.strip().splitlines():
        cxcmsid, msid = line.split()[:2]
        if invert:
            aliases[msid] = cxcmsid
        else:
            aliases[cxcmsid] = msid
    return aliases

ALIASES = {'simdiag': """
    RAMEXEC          3SDSWELF   SEA CSC Exectuting from RAM
    DSTACKPTR        3SDPSTKP   SEA Data Stack Ptr
    TSCEDGE          3SDTSEDG   TSC Tab Edge Detection Flags
    FAEDGE           3SDFAEDG   FA Tab Edge Detection Flags
    MJFTIME          3SDMAJFP   Major Frame Period Time Measured by SEA
    MRMDEST          3SDRMOVD   Most Recent Motor Move Destination
    TSCTABADC        3SDTSTSV   TSC Tab Position Sensor A/D converter
    FATABADC         3SDFATSV   FA Tab Position Sensor A/D Converter
    AGRNDADC         3SDAGV     Analog Ground A/D Converter Reading
    P15VADC          3SDP15V    +15V Power Supply A/D Converter Reading
    P5VADC           3SDP5V     +5V Power Supply A/D Converter Reading
    N15VADC          3SDM15V    -15V Power Supply A/D Converter Reading
    FLEXATEMPADC     3SDFLXAT   Flexture A Thermistor A/D Converter
    FLEXBTEMPADC     3SDFLXBT   Flexture B Thermistor A/D Converter
    FLEXCTEMPADC     3SDFLXCT   Flexture C Thermistor A/D Converter
    TSCMTRTEMPADC    3SDTSMT    TSC Motor Thermistor A/D Converter
    FAMTRTEMPADC     3SDFAMT    FA Motor Thermistor A/D Converter
    PSUTEMPADC       3SDPST     SEA Power Supply Thermistor A/D Converter
    BOXTEMPADC       3SDBOXT    SEA Box Thermistor A/D Converter
    RAMFAILADDR      3SDRMFAD   RAM Most Recent detected Fail Address
    TSCTABWID        3SDTSTBW   TSC Most Recent detected Tab Width
    FATABWID         3SDFATBW   FA Most Recent detected Tab Width
    SYNCLOSS         3SDSYRS    Process Reset Due Synchronization Loss
    WARMRESET        3SDWMRS    Processor Warm Reset
    TSCHISTO         3SDTSP     TSC Most Recent PWM Histogram
    FAHISTO          3SDFAP     FA Most Recent PWM Histogram
    INVCMDCODE       3SDINCOD   SEA Invalid CommandCode
    """,
           'sim_mrg': """
    TLMUPDATE    3SEATMUP   "Telemtry Update Flag"
    SEAIDENT     3SEAID     "SEA Identification Flag"
    SEARESET     3SEARSET   "SEA Reset Flag"
    PROMFAIL     3SEAROMF   "SEA PROM Checksum Flag"
    INVCMDGROUP  3SEAINCM   "SEA Invalid Command Group Flag"
    TSCMOVING    3TSCMOVE   "TSC In Motion Flag"
    FAMOVING     3FAMOVE    "FA In Motion Flag"
    FAPOS        3FAPOS     "FA Position"
    TSCPOS       3TSCPOS    "TSC Postion"
    PWMLEVEL     3MRMMXMV   "Max Power Motor Volt recent move"
    LDRTMECH     3LDRTMEK   "Last Detected Reference Mechanism Tab"
    LDRTNUM      3LDRTNO    "Last Detected Reference Tab Number"
    LDRTRELPOS   3LDRTPOS   "Last Detected Reference Relative Postion"
    FLEXATEMP    3FAFLAAT   "Flexture A Temperature"
    FLEXBTEMP    3FAFLBAT   "Flexture B Temperature"
    FLEXCTEMP    3FAFLCAT   "Flexture C Temperature"
    TSCMTRTEMP   3TRMTRAT   "TSC Motor Temperature"
    FAMTRTEMP    3FAMTRAT   "FA Motor Temperature"
    PSUTEMP      3FAPSAT    "SEA Power Supply Temperature"
    BOXTEMP      3FASEAAT   "SEA Box Temperature"
    STALLCNT     3SMOTSTL   "SEA Motor Stall Counter"
    TAB2AUTOPOS  3STAB2EN   "SEA Tab 2 Auto Position Update Status"
    MTRDRVRLY    3SMOTPEN   "SEA Motor Driver Power Relay status"
    MTRSELRLY    3SMOTSEL   "SEA Motor Selection Relay Status"
    HTRPWRRLY    3SHTREN    "SEA Heater Power Relay Status"
    RAMFAIL      3SEARAMF   "SEA RAM Failure Detected Flag"
    MTROVRCCNT   3SMOTOC    "Motor Drive Overcurrent Counter"
    PENDCMDCNT   3SPENDC    "SEA Pending Command Count"
    FLEXATSET    3SFLXAST   "Flexture A Temperature Setpoint"
    FLEXBTSET    3SFLXBST   "Flexture B Temperature Setpoint"
    FLEXCTSET    3SFLXCST   "Flexture C Temperature Setpoint"
    """,
           'hrc0ss': """
    TLEVART      2TLEV1RT
    VLEVART      2VLEV1RT
    SHEVART      2SHEV1RT
    TLEVART      2TLEV2RT
    VLEVART      2VLEV2RT
    SHEVART      2SHEV2RT
    """,
           'hrc0hk': """
    SCIDPREN:0,1,2,3,8,9,10 HRC_SS_HK_BAD
    P24CAST:7     224PCAST
    P15CAST:7     215PCAST
    N15CAST:7     215NCAST
    SPTPAST       2SPTPAST
    SPBPAST       2SPBPAST
    IMTPAST       2IMTPAST
    IMBPAST       2IMBPAST
    MTRSELCT:3    2NYMTAST
    MTRSELCT:4    2PYMTAST
    MTRSELCT:5    2CLMTAST
    MTRSELCT:6    2DRMTAST
    MTRSELCT:7    2ALMTAST
    MTRSTATR:0    2MSMDARS
    MTRSTATR:1    2MDIRAST
    MTRSTATR:2    2MSNBAMD
    MTRSTATR:3    2MSNAAMD
    MTRSTATR:4    2MSLBAMD
    MTRSTATR:5    2MSLAAMD
    MTRSTATR:6    2MSPRAMD
    MTRSTATR:7    2MSDRAMD
    MTRCMNDR:0    2MCMDARS
    MTRCMNDR:2    2MCNBAMD
    MTRCMNDR:3    2MCNAAMD
    MTRCMNDR:4    2MCLBAMD
    MTRCMNDR:5    2MCLAAMD
    MTRCMNDR:6    2MCPRAMD
    MTRCMNDR:7    2MDRVAST
    SCTHAST       2SCTHAST
    MTRITMP:1     2SMOIAST
    MTRITMP:2     2SMOTAST
    MTRITMP:5     2DROTAST
    MTRITMP:6     2DROIAST
    MLSWENBL:3    2SFLGAST
    MLSWENBL:4    2OSLSAST
    MLSWENBL:5    2OPLSAST
    MLSWENBL:6    2CSLSAST
    MLSWENBL:7    2CPLSAST
    MLSWSTAT:2    2OSLSADT
    MLSWSTAT:3    2OSLSAAC
    MLSWSTAT:4    2OPLSAAC
    MLSWSTAT:5    2CSLSADT
    MLSWSTAT:6    2CSLSAAC
    MLSWSTAT:7    2CPLSAAC
    FCPUAST       2FCPUAST
    FCPVAST       2FCPVAST
    CBHUAST       2CBHUAST
    CBLUAST       2CBLUAST
    CBHVAST       2CBHVAST
    CBLVAST       2CBLVAST
    WDTHAST       2WDTHAST
    SCIDPREN:4    2CLMDAST
    SCIDPREN:5    2FIFOAVR
    SCIDPREN:6    2OBNLASL
    SCIDPREN:7    2SPMDASL
    SCIDPREN:11   2EBLKAVR
    SCIDPREN:12   2CBLKAVR
    SCIDPREN:13   2ULDIAVR
    SCIDPREN:14   2WDTHAVR
    SCIDPREN:15   2SHLDAVR
    HVPSSTAT:0    2SPONST
    HVPSSTAT:1    2SPCLST
    HVPSSTAT:2    2S1ONST
    HVPSSTAT:3    2IMONST
    HVPSSTAT:4    2IMCLST
    HVPSSTAT:5    2S2ONST
    S1HVST        2S1HVST
    S2HVST        2S2HVST
    C05PALV       2C05PALV
    C15PALV       2C15PALV
    C15NALV       2C15NALV
    C24PALV       2C24PALV
    IMHVLV        2IMHVLV
    IMHBLV        2IMHBLV
    SPHVLV        2SPHVLV
    SPHBLV        2SPHBLV
    S1HVLV        2S1HVLV
    S2HVLV        2S2HVLV
    PRBSCR        2PRBSCR
    PRBSVL        2PRBSVL
    ULDIALV       2ULDIALV
    LLDIALV       2LLDIALV
    FEPRATM       2FEPRATM
    CALPALV       2CALPALV
    GRDVALV       2GRDVALV
    RSRFALV       2RSRFALV
    SPINATM       2SPINATM
    IMINATM       2IMINATM
    LVPLATM       2LVPLATM
    SPHVATM       2SPHVATM
    IMHVATM       2IMHVATM
    SMTRATM       2SMTRATM
    FE00ATM       2FE00ATM
    CE00ATM       2CE00ATM
    CE01ATM       2CE01ATM
    """,
           }


CXC_TO_MSID = {key: parse_alias_str(val) for key, val in ALIASES.items()}
MSID_TO_CXC = {key: parse_alias_str(val, invert=True) for key, val in ALIASES.items()}


def sim_mrg(dat):
    """
    Custom converter for SIM_MRG.

    There is a bug in CXCDS L0 SIM decom wherein the 3LDRTMEK MSID is
    incorrectly assigned (TSC and FA are reversed).  The calibration
    of 3LDRTPOS from steps to mm is then also wrong because it uses
    the FA conversion instead of TSC.

    This function fixes 3LDRTMEK, then backs out the (incorrect) 3LDRTPOS
    steps to mm conversion and re-does it correctly using the TSC conversion.
    Note that 3LDRTMEK is (by virtue of the way mission operations run)
    always "TSC".
    """
    # Start with the generic converter
    out = generic_converter(aliases=CXC_TO_MSID['sim_mrg'])(dat)

    # Now do the fixes.  FOT mech has stated that 3LDRTMEK is always 'FA'
    # in practice.
    bad = out['3LDRTMEK'] == 'FA '
    if np.count_nonzero(bad):
        out['3LDRTMEK'][bad] = 'TSC'
        pos_tsc_steps = units.converters['mm', 'FASTEP'](out['3LDRTPOS'][bad])
        out['3LDRTPOS'][bad] = units.converters['TSCSTEP', 'mm'](pos_tsc_steps)

    return out


simdiag = generic_converter(aliases=CXC_TO_MSID['simdiag'])
hrc0ss = generic_converter2(MSID_TO_CXC['hrc0ss'])


def hrc0hk(dat):
    # Read the data and allow for missing columns in input L0 HK file.
    default_dtypes = {'2CE00ATM': 'f4',
                      '2CE01ATM': 'f4'}
    out = generic_converter2(MSID_TO_CXC['hrc0hk'], default_dtypes)(dat)

    # Set all HRC HK data columns to bad quality where HRC_SS_HK_BAD is not zero
    # First three columns are TIME, QUALITY, and HRC_SS_HK_BAD -- do not filter these.
    bad = out['HRC_SS_HK_BAD'] > 0
    if np.any(bad):
        out['QUALITY'][bad, 3:] = True
        logger.info('Setting {} readouts of all HRC HK telem to bad quality (bad SCIDPREN)'
                    .format(np.count_nonzero(bad)))

    # Detect the secondary-science byte-shift anomaly by finding out-of-range 2SMTRATM values.
    # For those bad frames:
    # - Set bit 10 (from LSB) of HRC_SS_HK_BAD
    # - Set all analog MSIDs (2C05PALV and later in the list) to bad quality
    bad = (out['2SMTRATM'] < -20) | (out['2SMTRATM'] > 50)
    if np.any(bad):
        out['HRC_SS_HK_BAD'][bad] |= 2 ** 10  # 1024
        analogs_index0 = list(out.dtype.names).index('2C05PALV')
        out['QUALITY'][bad, analogs_index0:] = True
        logger.info('Setting {} readouts of analog HRC HK telem to bad quality (bad 2SMTRATM)'
                    .format(np.count_nonzero(bad)))

    return out


def obc4eng(dat):
    """
    At 2014:342:XX:XX:XX, patch PR-361 was applied which transitioned 41 OBA thermistors to
    read out in wide-mode.  After this time the data in the listed OOBTHRxx MSIDs became
    invalid while the OOBTHRxx_WIDE MSIDs became valid.  This converter simply copies the
    *_WIDE values to the original MSIDs after the time of patch activation.  The *_WIDE
    MSIDs are not available in the eng archive (by the _WIDE names).
    """
    # MSIDs OOBTHR<msid_num> that went to _WIDE after the patch, which was done in parts A
    # and B.
    msid_nums = {'a': '08 09 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29'.split(),
                 'b': '30 31 33 34 35 36 37 38 39 40 41 44 45 46 49 50 51 52 53 54'.split(),
                 'c': '02 03 04 05 06 07'.split()
                 }

    # Convert using the baseline converter
    out = numpy_converter(dat)

    # The patch times below correspond to roughly the middle of the major frame where
    # patches A and B were applied, respectively.
    patch_times = {'a': DateTime('2014:342:16:29:30').secs,
                   'b': DateTime('2014:342:16:32:45').secs,
                   'c': DateTime('2017:312:16:11:16').secs}

    for patch in ('a', 'b', 'c'):
        # Set a mask defining times after the activation of wide-range telemetry in PR-361
        mask = out['TIME'] > patch_times[patch]
        if np.any(mask):
            for msid_num in msid_nums[patch]:
                msid = 'OOBTHR' + msid_num
                msid_wide = msid + '_WIDE'
                print('Fixing MSID {}'.format(msid))
                out[msid][mask] = out[msid_wide][mask]

                q_index = quality_index(out, msid)
                q_index_wide = quality_index(out, msid_wide)
                out['QUALITY'][mask, q_index] = out['QUALITY'][mask, q_index_wide]

    return out


def tel2eng(dat):
    """
    At 2014:342:XX:XX:XX, patch PR-361 was applied which transitioned 41 OBA thermistors to
    read out in wide-mode.  As 4OAVOBAT is an average of all these MSIDs and calculated on board,
    only the wide version of this MSID is valid after this patch is applied.

    This converter simply copies the 4OAVOBAT_WIDE values after the time of patch activation to
    4OAVOBAT.  4OAVOBAT_WIDE is not available in the eng archive (by the _WIDE name).
    """

    # Convert using the baseline converter
    out = numpy_converter(dat)

    # 4OAVOBAT is modified by both patches since it is an average of MSIDs in both parts of the
    # patch. Use the second time value as this is when the process is complete. See obc4eng() for
    # both times and further details.
    patch_time = DateTime('2014:342:16:32:45').secs

    mask = out['TIME'] > patch_time
    if np.any(mask):
        print('Fixing MSID 4OAVOBAT')
        out['4OAVOBAT'][mask] = out['4OAVOBAT_WIDE'][mask]

        q_index = quality_index(out, '4OAVOBAT')
        q_index_wide = quality_index(out, '4OAVOBAT_WIDE')
        out['QUALITY'][mask, q_index] = out['QUALITY'][mask, q_index_wide]

    return out


def acisdeahk(dat):
    """Take the archive ACIS-0 DEA HKP data and convert to a format that is
    consistent with normal eng0 files.  ACIS-0 housekeeping has data stored
    in query-records, one row per DEA statistic query.  Gather all the
    time-synced queries corresponding to columns for the acis0hkp table
    and put into a single row.  Write out to temp files and modify self->{arch_files}.
    """

    logger.info('Converting acisdeahk data to eng0 format')

    cols = _get_deahk_cols()
    col_query_ids = tuple(x['query_id'] for x in cols)
    col_names = tuple(x['name'].upper() for x in cols)

    # Filter only entries with ccd_id >= 10 which indicates data from the I/F control
    dat = pyfits_to_recarray(dat)
    rows = dat[dat['CCD_ID'] >= 10]
    if len(rows) == 0:
        raise NoValidDataError()

    # Go through input data one row (query) at a time and assemble contemporaneous
    # queries into a single row with a column for each query value.
    # Collect each assembled row into %data_out for writing to a FITS bin table
    block_idxs = 1 + numpy.flatnonzero(numpy.abs(rows['TIME'][1:] - rows['TIME'][:-1]) > 0.001)
    block_idxs = numpy.hstack([[0], block_idxs, [len(rows)]])
    query_val_tus = rows['QUERY_VAL_TU']
    query_vals = rows['QUERY_VAL']
    query_ids = rows['QUERY_ID']
    times = rows['TIME']

    outs = []
    for i0, i1 in zip(block_idxs[:-1], block_idxs[1:]):
        # Map query_id to an index into rows for each row in the block
        id_idxs = dict((query_ids[i], i) for i in range(i0, i1))

        # Make tuples of the values and qual flags corresponding to each DEAHK_COLUMN
        bads = tuple(query_id not in id_idxs for query_id in col_query_ids)
        vals = tuple((0.0 if bad else query_vals[id_idxs[query_id]])
                     for bad, query_id in zip(bads, col_query_ids))
        val_tus = tuple((0 if bad else query_val_tus[id_idxs[query_id]])
                     for bad, query_id in zip(bads, col_query_ids))

        # Now have another pass at finding bad values.  Take these out now so the
        # 5min and daily stats are not frequently corrupted.
        bads = tuple(True if (val_tu == 65535 or numpy.isnan(val)) else bad
                     for bad, val, val_tu in zip(bads, vals, val_tus))

        quality = (False, False) + bads
        outs.append((times[i0], quality) + vals)

    dtype = [('TIME', numpy.float64),
              ('QUALITY', numpy.bool, (len(col_names) + 2,))]
    dtype += [(col_name, numpy.float32) for col_name in col_names]

    return numpy.rec.fromrecords(outs, dtype=dtype)

def _get_deahk_cols():
    out = [
        {
            "query_id": 1,
            "name": "tmp_bep_pcb",
            "unit": "K",
            "descr": "DPA Thermistor 1 - BEP PC Board"
        },
        {
            "query_id": 2,
            "name": "tmp_bep_osc",
            "unit": "K",
            "descr": "DPA Thermistor 2 - BEP Oscillator"
        },
        {
            "query_id": 3,
            "name": "tmp_fep0_mong",
            "unit": "K",
            "descr": "DPA Thermistor 3 - FEP 0 Mongoose"
        },
        {
            "query_id": 4,
            "name": "tmp_fep0_pcb",
            "unit": "K",
            "descr": "DPA Thermistor 4 - FEP 0 PC Board"
        },
        {
            "query_id": 5,
            "name": "tmp_fep0_actel",
            "unit": "K",
            "descr": "DPA Thermistor 5 - FEP 0 ACTEL"
        },
        {
            "query_id": 6,
            "name": "tmp_fep0_ram",
            "unit": "K",
            "descr": "DPA Thermistor 6 - FEP 0 RAM"
        },
        {
            "query_id": 7,
            "name": "tmp_fep0_fb",
            "unit": "K",
            "descr": "DPA Thermistor 7 - FEP 0 Frame Buf"
        },
        {
            "query_id": 8,
            "name": "tmp_fep1_mong",
            "unit": "K",
            "descr": "DPA Thermistor 8 - FEP 1 Mongoose"
        },
        {
            "query_id": 9,
            "name": "tmp_fep1_pcb",
            "unit": "K",
            "descr": "DPA Thermistor 9 - FEP 1 PC Board"
        },
        {
            "query_id": 10,
            "name": "tmp_fep1_actel",
            "unit": "K",
            "descr": "DPA Thermistor 10 - FEP 1 ACTEL"
        },
        {
            "query_id": 11,
            "name": "tmp_fep1_ram",
            "unit": "K",
            "descr": "DPA Thermistor 11 - FEP 1 RAM"
        },
        {
            "query_id": 12,
            "name": "tmp_fep1_fb",
            "unit": "K",
            "descr": "DPA Thermistor 12 - FEP 1 Frame Buf"
        },
        {
            "query_id": 15,
            "name": "fptemp_12",
            "unit": "K",
            "descr": "Focal Plane Temp. Board 12"
        },
        {
            "query_id": 16,
            "name": "fptemp_11",
            "unit": "K",
            "descr": "Focal Plane Temp. Board 11"
        },
        {
            "query_id": 17,
            "name": "dpagndref1",
            "unit": "V",
            "descr": "DPA Ground Reference 1"
        },
        {
            "query_id": 18,
            "name": "dpa5vhka",
            "unit": "V",
            "descr": "DPA 5V Housekeeping A"
        },
        {
            "query_id": 19,
            "name": "dpagndref2",
            "unit": "V",
            "descr": "DPA Ground Reference 2"
        },
        {
            "query_id": 20,
            "name": "dpa5vhkb",
            "unit": "V",
            "descr": "DPA 5V Housekeeping B"
        },
        {
            "query_id": 25,
            "name": "dea28volta",
            "unit": "V",
            "descr": "Primary Raw DEA 28V DC"
        },
        {
            "query_id": 26,
            "name": "dea24volta",
            "unit": "V",
            "descr": "Primary Raw DEA 24V DC"
        },
        {
            "query_id": 27,
            "name": "deam15volta",
            "unit": "V",
            "descr": "Primary Raw DEA -15.5V"
        },
        {
            "query_id": 28,
            "name": "deap15volta",
            "unit": "V",
            "descr": "Primary Raw DEA +15.5V"
        },
        {
            "query_id": 29,
            "name": "deam6volta",
            "unit": "V",
            "descr": "Primary Raw DEA -6V DC"
        },
        {
            "query_id": 30,
            "name": "deap6volta",
            "unit": "V",
            "descr": "Primary Raw DEA +6V DC"
        },
        {
            "query_id": 31,
            "name": "rad_pcb_a",
            "descr": "Relative Dose Rad. Monitor Side A"
        },
        {
            "query_id": 32,
            "name": "gnd_1",
            "unit": "V",
            "descr": "Interface Ground Reference"
        },
        {
            "query_id": 33,
            "name": "dea28voltb",
            "unit": "V",
            "descr": "Backup Raw DEA 28V DC"
        },
        {
            "query_id": 34,
            "name": "dea24voltb",
            "unit": "V",
            "descr": "Backup DEA 24V DC"
        },
        {
            "query_id": 35,
            "name": "deam15voltb",
            "unit": "V",
            "descr": "Backup DEA -15.5V DC"
        },
        {
            "query_id": 36,
            "name": "deap15voltb",
            "unit": "V",
            "descr": "Backup DEA +15.5V DC"
        },
        {
            "query_id": 37,
            "name": "deam6voltb",
            "unit": "V",
            "descr": "Backup DEA -6V DC"
        },
        {
            "query_id": 38,
            "name": "deap6voltb",
            "unit": "V",
            "descr": "Backup DEA +6V DC"
        },
        {
            "query_id": 39,
            "name": "rad_pcb_b",
            "descr": "Relative Dose Rad. Monitor Side B"
        },
        {
            "query_id": 40,
            "name": "gnd_2",
            "unit": "V",
            "descr": "Ground"
        }
        ]
    return out

def pyfits_to_recarray(dat):
    dtypes = []
    colnames = dat.dtype.names
    for colname in colnames:
        col = dat.field(colname)
        if col.dtype.isnative:
            dtype = (colname, col.dtype)
        else:
            dtype = (colname, col.dtype.type)
        if len(col.shape) > 1:
            dtype = dtype + tuple(col.shape[1:])
        dtypes.append(dtype)

    # Now define a new recarray and copy the original data
    # Note: could use numpy.empty to generate a structured array.
    out = numpy.recarray(len(dat), dtype=dtypes)
    for colname in colnames:
        out[colname][:] = dat.field(colname)

    return out
