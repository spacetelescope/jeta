""" A tool to generate sample files for ingest.

These files can be used to test algorithms with more
realistic data.

"""
import argparse
from astropy.time import Time


def get_options(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-of-files",
        type=int,
        default=1,
        help="The number of ingest files to generate for testing."
    )

    parser.add_argument(
        "--date-now",
        default=Time.now(),
        help="Set start time for the first file for testing (default=NOW)"
    )

    return parser.parse_args(args)


def main():
    pass
