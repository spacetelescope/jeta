# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fetch telemetry from the Ska engineering archive.

Examples
========

  # Get full-resolution TEPHIN, AOPCADMD for 30 days, and save as telem.zip
  % ska_fetch TEPHIN AOPCADMD --start=2013:001 --stop=2013:030 --sampling=5min \\
              --time-format=greta --outfile=telem.zip

  # Get daily temps since 2000, removing times within 100000 seconds of safe- or normal- sun
  % ska_fetch TEPHIN TCYLAFT6 --start 2000:001 --sampling=daily --outfile=tephin.zip \\
              --remove-events='safe_suns[pad=100000] | normal_suns[pad=100000]'

  # Get daily IRU-2 temps since 2004, removing know LTT bad times
  % ska_fetch AIRU2BT --start 2004:001 --sampling=daily --outfile=airu2bt.zip \\
              --remove-events='ltt_bads[msid="AIRU2BT"]'

Arguments
=========
"""
from __future__ import print_function, division, absolute_import

import ast
import argparse
import re
import shlex
from itertools import count
from six.moves import zip
import six

import numpy as np

from Chandra.Time import DateTime
from . import fetch, utils


def sanitize_event_expression(expr):
    """
    Take an event expression from the --remove-intervals or --select-intervals
    command line args and do two things:

    - Fully parse and make sure it matches the limited allowed syntax so that
      it can be later eval'd safely.
    - Change e.g. "ltt_bads[pad=800, msid='airu2bt']" to
      "events.ltt_bads(pad=800, msid='airu2bt')" for eval in module context.
    """
    # First tokenize
    seps = '()~|&'
    expr = re.sub('\s+', '', expr)
    words = [''] + list(shlex.shlex(expr)) + ['']

    tokens = []
    for word in words:
        if word in seps or word == '':
            tokens.append("SEP")
        elif re.match(r'\w+$', word):
            try:
                ast.literal_eval(word)
                tokens.append("LITERAL")
            except:
                tokens.append("SYMBOL")
        elif word == '=':
            tokens.append("EQUAL")
        elif word == '[':
            tokens.append("LBRACE")
        elif word == ']':
            tokens.append("RBRACE")
        elif word == ',':
            tokens.append("COMMA")
        else:
            try:
                ast.literal_eval(word)
                tokens.append("LITERAL")
            except:
                raise ValueError('Cannot identify word {!r}'.format(word))

    # Now check syntax and do substitutions where needed
    in_arg_list = False
    for i, p_token, token, n_token in zip(count(1), tokens, tokens[1:], tokens[2:]):
        if token == "SEP":
            if words[i] in '|&':
                words[i] = ' {} '.format(words[i])
            continue

        if token == "LBRACE":
            if in_arg_list:
                raise ValueError('"LBRACE" within arg list')
            in_arg_list = True
            words[i] = '('

        elif token == "RBRACE":
            if not in_arg_list:
                raise ValueError('"RBRACE" outside arg list')
            in_arg_list = False
            words[i] = ')'

        elif token == "SYMBOL":
            if not in_arg_list:
                words[i] = 'events.' + words[i]
                if n_token not in ("SEP", "LBRACE"):
                    raise ValueError('"SYMBOL" not followed by "SEP" or "LBRACE"')
            else:
                if n_token != "EQUAL":
                    raise ValueError('KWARG not followed by "EQUAL"')

        elif token == "EQUAL":
            if p_token != "SYMBOL" or n_token != "LITERAL":
                raise ValueError('SYMBOL = LITERAL syntax not found')

        elif token == "LITERAL":
            if p_token != "EQUAL" or n_token not in ("RBRACE", "COMMA"):
                raise ValueError('Bad "LITERAL" placement')

        elif token == "COMMA":
            if n_token != "SYMBOL":
                raise ValueError('COMMA must be followed by SYMBOL')
            words[i] = ', '

    return ''.join(words)


def msidset_resample(msidset, dt):
    """
    Resample the ``msidset`` in-place to a common time basis, starting at
    ``msidset.tstart`` and stepping by ``dt``.  This assumes an unfiltered MSIDset, and
    returns a filtered MSIDset.
    """
    tstart = (msidset.tstart // dt) * dt
    times = np.arange((msidset.tstop - tstart) // dt) * dt + tstart
    msidset.interpolate(times=times, filter_bad=False)
    common_bads = np.zeros(len(msidset.times), dtype=bool)
    for msid in msidset.values():
        if msid.bads is not None:  # 5min and daily stats have no bad values
            common_bads |= msid.bads

    # Apply the common bads array and filter out these bad values
    for msid in msidset.values():
        msid.bads = common_bads
        msid.filter_bad()
    msidset.times = msidset.times[~common_bads]


def get_queryset(expr):
    """
    Get query set for ``expr`` python-ish expression, e.g. (manvrs[bad=200] & radzones).
    Using [] instead of () makes parsing easier.
    """
    from kadi import events

    queryset_expr = sanitize_event_expression(expr)
    return eval(queryset_expr)


def get_telem(msids, start=None, stop=None, sampling='full', unit_system='eng',
              interpolate_dt=None, remove_events=None, select_events=None,
              time_format=None, outfile=None, quiet=False,
              max_fetch_Mb=None, max_output_Mb=None):
    """
    High-level routine to get telemetry for one or more MSIDs and perform
    common post-processing functions.

    This is a non-public version that really does the work.  The public interface
    is fetch.get_telem(), which is a thin wrapper for this.  (Trying to factor code
    out to separate modules and keep import times down).  See get_telem() for param
    docs.
    """
    # Set up output logging
    from pyyaks.logger import get_logger
    logger = get_logger(name='Ska.engarchive.get_telem', level=(100 if quiet else -100))

    # Set defaults and translate to fetch keywords
    stop = DateTime(stop)
    start = stop - 30 if start is None else DateTime(start)
    stat = None if sampling == 'full' else sampling
    filter_bad = interpolate_dt is None
    if isinstance(msids, six.string_types):
        msids = [msids]

    logger.info('Fetching {}-resolution data for MSIDS={}\n  from {} to {}'
                .format(sampling, msids, start.date, stop.date))

    fetch.set_units(unit_system)

    # Make sure that the dataset being fetched is reasonable (if checking requested)
    if max_fetch_Mb is not None or max_output_Mb is not None:
        fetch_Mb, output_Mb = utils.get_fetch_size(msids, start, stop, stat=stat,
                                                   interpolate_dt=interpolate_dt, fast=True)
        if max_fetch_Mb is not None and fetch_Mb > max_fetch_Mb:
            raise MemoryError('Requested fetch requires {:.2f} Mb vs. limit of {:.2f} Mb'
                              .format(fetch_Mb, max_fetch_Mb))
        # If outputting to a file then check output size
        if outfile and max_output_Mb is not None and output_Mb > max_output_Mb:
            raise MemoryError('Requested fetch (interpolated) requires {:.2f} Mb '
                              'vs. limit of {:.2f} Mb'
                              .format(output_Mb, max_output_Mb))

    dat = fetch.MSIDset(msids, start, stop, stat=stat, filter_bad=filter_bad)

    if interpolate_dt is not None:
        logger.info('Interpolating at {} second intervals'.format(interpolate_dt))
        msidset_resample(dat, interpolate_dt)

    if remove_events is not None:
        logger.info('Removing events: {}'.format(remove_events))
        queryset = get_queryset(remove_events)
        for msid in dat:
            dat[msid].remove_intervals(queryset)

    if select_events is not None:
        logger.info('Selecting events: {}'.format(select_events))
        queryset = get_queryset(select_events)
        for msid in dat:
            dat[msid].select_intervals(queryset)

    if time_format not in (None, 'secs'):
        for dat_msid in dat.values():
            dat_msid.times = getattr(DateTime(dat_msid.times, format='secs'), time_format)

    if outfile:
        logger.info('Writing data to {}'.format(outfile))
        dat.write_zip(outfile)

    return dat


def get_opt():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start',
                        type=str,
                        help='Start time for data fetch (default=<stop> - 30 days)')

    parser.add_argument('--stop',
                        type=str,
                        help='Stop time for data fetch (default=NOW)')

    parser.add_argument('--sampling',
                        type=str,
                        default='5min',
                        help='Data sampling (full|5min|daily) (default=5min)')

    parser.add_argument('--unit-system',
                        type=str,
                        default='eng',
                        help='Unit system for data (eng|sci|cxc) (default=eng)')

    parser.add_argument('--interpolate-dt',
                        type=float,
                        help='Interpolate to uniform time steps (secs, default=None)')

    parser.add_argument('--remove-events',
                        type=str,
                        help='Remove kadi events expression (default=None)')

    parser.add_argument('--select-events',
                        type=str,
                        help='Select kadi events expression (default=None)')

    parser.add_argument('--time-format',
                        type=str,
                        help='Output time format (secs|date|greta|jd|frac_year|...)')

    parser.add_argument('--outfile',
                        default='fetch.zip',
                        type=str,
                        help='Output file name (default=fetch.zip)')

    parser.add_argument('--quiet',
                        action='store_true',
                        help='Suppress run-time logging output')

    parser.add_argument('--max-fetch-Mb',
                        default=1000.0,
                        type=float,
                        help='Max allowed memory (Mb) for fetching (default=1000)')

    parser.add_argument('--max-output-Mb',
                        default=100.0,
                        type=float,
                        help='Max allowed memory (Mb) for file output (default=100)')

    parser.add_argument('msids',
                        metavar='MSID',
                        type=str,
                        nargs='+',
                        help='MSID to fetch')

    opt = parser.parse_args()
    return opt


def main():
    opt = get_opt()
    try:
        get_telem(**vars(opt))
    except MemoryError as err:
        print('\n'.join(['',
                         '*' * 80,
                         'ERROR: {}'.format(err),
                         '*' * 80, '']))
    except Exception as err:
        print('\n'.join(['',
                         '*' * 80,
                         'ERROR: {}'.format(err),
                         'If necessary report the following traceback to aca@head.cfa.harvard.edu',
                         '*' * 80, '']))
        raise
