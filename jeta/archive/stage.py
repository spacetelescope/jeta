import os
import pickle
import sqlite3
import glob
import ntpath
import shutil
import json

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


def create_collection(name, file_collection=[], description=""):

    _collection = f"{STAGING_DIRECTORY}/{name}"
    if os.path.exists(_collection):
        raise IOError(f"Cannot create collection {_collection} already exists.")
    else:
        os.mkdir(_collection)
        for f in file_collection:
            shutil.move(f"{STAGING_DIRECTORY}/{f}", f"{_collection}/{f}")

    return 0
