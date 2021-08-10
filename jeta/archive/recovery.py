def fix_state_code(state_code):
    """
    Return a version of ``state_code`` that has only alphanumeric chars.  This
    can be used as a column name, unlike e.g. "n_+1/2".  Since this gets called
    in an inner loop cache the result.
    """
    try:
        out = _fix_state_code_cache[state_code]
    except KeyError:
        out = state_code
        for sub_in, sub_out in ((r'\+', 'PLUS_'),
                                (r'\-', 'MINUS_'),
                                (r'>', '_GREATER_'),
                                (r'/', '_DIV_')):
            out = re.sub(sub_in, sub_out, out)
        _fix_state_code_cache[state_code] = out

    return out


def fix_misorders(filetype):
    """Fix problems in the eng archive where archive files were ingested out of
    time order.  This results in a non-monotonic times in the MSID hdf5 files
    and subsequently corrupts the stats files.  This routine looks for
    discontinuities in rowstart assuming filename ordering and swaps neighbors.
    One needs to verify in advance (--dry-run --fix-misorders --content ...)
    that this will be an adequate fix.
    Example::
      update_archive.py --dry-run --fix-misorders --content misc3eng
      update_archive.py --fix-misorders --content misc3eng >& fix_misc3.log
      update_archive.py --content misc3eng --max-lookback-time 100 >>& fix_misc3.log
    In the --dry-run it is important to verify that the gap is really just from
    two mis-ordered files that can be swapped.  Look at the rowstart,rowstop values
    in the filename-ordered list.
    :param filetype: filetype
    :returns: minimum time for all misorders found
    """
    colnames = pickle.load(open(msid_files['colnames'].abs, 'rb'))

    # Setup db handle with autocommit=False so that error along the way aborts insert transactions
    db = Ska.DBI.DBI(dbi='sqlite', server=msid_files['archfiles'].abs, autocommit=False)

    # Get misordered archive files
    archfiles = db.fetchall('SELECT * FROM archfiles order by filename')
    bads = archfiles['rowstart'][1:] - archfiles['rowstart'][:-1] < 0

    if not np.any(bads):
        logger.info('No misorders')
        return

    for bad in np.flatnonzero(bads):
        i2_0, i1_0 = archfiles['rowstart'][bad:bad + 2]
        i2_1, i1_1 = archfiles['rowstop'][bad:bad + 2]

        # Update hdf5 file for each column (MSIDs + TIME, MJF, etc)
        for colname in colnames:
            ft['msid'] = colname
            logger.info('Fixing %s', msid_files['msid'].abs)
            if not opt.dry_run:
                filepath = msid_files['mnemonic_value'].abs
                h5 = tables.open_file(filepath, mode='a')
                #h5 = tables.open_file(msid_files['msid'].abs, mode='a')
                hrd = h5.root.data
                hrq = h5.root.quality

                hrd1 = hrd[i1_0:i1_1]
                hrd2 = hrd[i2_0:i2_1]
                hrd[i1_0:i1_0 + len(hrd2)] = hrd2
                hrd[i1_0 + len(hrd2): i2_1] = hrd1

                hrq1 = hrq[i1_0:i1_1]
                hrq2 = hrq[i2_0:i2_1]
                hrq[i1_0:i1_0 + len(hrq2)] = hrq2
                hrq[i1_0 + len(hrq2): i2_1] = hrq1

                h5.close()

        # Update the archfiles table
        cmd = 'UPDATE archfiles SET '
        cols = ['rowstart', 'rowstop']
        cmd += ', '.join(['%s=?' % x for x in cols])
        cmd += ' WHERE filename=?'
        rowstart1 = i1_0
        rowstop1 = rowstart1 + i2_1 - i2_0
        rowstart2 = rowstop1 + 1
        rowstop2 = i2_1
        vals1 = [rowstart1, rowstop1, archfiles['filename'][bad]]
        vals2 = [rowstart2, rowstop2, archfiles['filename'][bad + 1]]
        logger.info('Running %s %s', cmd, vals1)
        logger.info('Running %s %s', cmd, vals2)

        logger.info('Swapping rows %s for %s', [i1_0, i1_1, i2_0, i2_1], filetype.content)
        logger.info('%s', archfiles[bad - 3:bad + 5])
        logger.info('')

        if not opt.dry_run:
            db.execute(cmd, [x.tolist() for x in vals1])
            db.execute(cmd, [x.tolist() for x in vals2])
            db.commit()

    return np.min(archfiles['tstart'][bads])