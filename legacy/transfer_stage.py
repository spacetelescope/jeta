#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Perform file transfer functions to get files over to GRETA network.

This is normally run as a cron job.
"""

if __name__ == '__main__':
    from Ska.engarchive import transfer_stage
    transfer_stage.main()
