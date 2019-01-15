#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Transfer stage files from HEAD network to OCC GRETA network via ftp server lucky.
"""

import re, os, sys
import time
import optparse
import tarfile
import shutil

from Chandra.Time import DateTime
import Ska.ftp
import Ska.engarchive.file_defs as file_defs
import Ska.File
import pyyaks.logger
import pyyaks.context


opt = None
logger = None


def get_options():
    parser = optparse.OptionParser()
    parser.add_option("--data-root",
                      help="Engineering archive root directory for MSID and arch files")
    parser.add_option("--ftp-dir",
                      default='eng_archive',
                      help="ftp directory name (default='eng_archive')")
    parser.add_option("--timeout",
                      default=6 * 3600,
                      type="int",
                      help="Tool timeout before quitting")
    parser.add_option("--sleep-time",
                      default=300,
                      type="int",
                      help="Time to sleep while waiting for files on lucky")
    parser.add_option("--occ",
                      action="store_true",
                      help="Running on the OCC GRETA network (no arc5gl)")
    return parser.parse_args()


def transfer_stage_to_lucky():
    # Make a tarfile of everything in stage directory
    tarname = 'stage_{0:d}.tar'.format(int(DateTime().secs))
    logger.info('Making tarfile {0}'.format(tarname))
    tar = tarfile.open(name=tarname, mode='w')
    tar.add('stage')
    tar.close()

    # Put the tarfile on lucky.  First put it at the top level, then when
    # complete move it into a subdir eng_archive.  This lets the OCC side
    # just watch for fully-uploaded files in that directory.
    logger.info('ftp to lucky')
    ftp = Ska.ftp.SFTP('lucky', logger=logger)
    ftp.cd('/home/taldcroft')
    files = ftp.ls()
    if opt.ftp_dir not in files:
        logger.info('mkdir {}'.format(opt.ftp_dir))
        ftp.mkdir(opt.ftp_dir)
    logger.info('put {0}'.format(tarname))
    ftp.put(tarname)
    logger.info('rename {0} {1}/{0}'.format(tarname, opt.ftp_dir))
    ftp.rename(tarname, '{}/{}'.format(opt.ftp_dir, tarname))
    ftp.close()

    logger.info('unlink {0}'.format(tarname))
    os.unlink(tarname)
    logger.info('rmtree stage')
    shutil.rmtree('stage')


def transfer_lucky_to_stage():
    """
    Get tarfile(s) from lucky and untar into stage area.  This is all done from
    the archive root directory.
    """

    # Open lucky ftp connection and watch for tarfile(s) in '/taldcroft/eng_archive'
    logger.info('ftp to lucky')
    ftp = Ska.ftp.SFTP('lucky', logger=logger)
    ftp.cd('/home/taldcroft')
    files = ftp.ls()
    if opt.ftp_dir not in files:
        logger.info('mkdir {}'.format(opt.ftp_dir))
        ftp.mkdir(opt.ftp_dir)
    ftp.cd(opt.ftp_dir)
    for _ in range(opt.timeout / opt.sleep_time):
        logger.info('Directory: {}'.format(ftp.ftp.getcwd()))
        logger.info('Files: {}'.format(ftp.ls()))
        files = [x for x in ftp.ls() if re.match('stage_\d+\.tar', x)]
        if files:
            break
        time.sleep(opt.sleep_time)
        logger.info('Waiting {0} seconds for archive files to appear...'.format(opt.sleep_time))
    else:
        logger.info('No tarfiles found before timeout')
        ftp.close()
        sys.exit(1)

    # For each tarfile:
    # - get file by ftp
    # - untar
    # - unlink local tarfile
    # - delete tarfile on ftp
    for tarname in sorted(files):
        logger.info('Getting tarfile {0} from lucky'.format(tarname))
        ftp.get(tarname)
        logger.info('Extracting tarfile {0}'.format(tarname))
        tar = tarfile.open(name=tarname, mode='r')
        tar.extractall()
        tar.close()

        logger.info('Unlinking local {0}'.format(tarname))
        os.unlink(tarname)
        logger.info('Deleting ftp {0}'.format(tarname))
        ftp.delete(tarname)

    ftp.close()


def main():
    global opt
    global logger

    opt, args = get_options()

    loglevel = pyyaks.logger.VERBOSE
    logger = pyyaks.logger.get_logger(name='transfer_stage', level=loglevel,
                                      format="%(asctime)s %(message)s")
    arch_files = pyyaks.context.ContextDict('arch_files', basedir=opt.data_root)
    arch_files.update(file_defs.arch_files)

    with Ska.File.chdir(arch_files['rootdir'].abs):
        if opt.occ:
            transfer_lucky_to_stage()
        else:
            transfer_stage_to_lucky()
