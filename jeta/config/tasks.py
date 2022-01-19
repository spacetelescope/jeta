import os
import datetime
import subprocess

import numpy as np
from astropy.time import Time

from jeta.staging.manage import get_staged_files

from config.celery import app

@app.task
def debug_task():
    print(f"DEBUG TASK: {Time(datetime.datetime.now(), format='datetime').yday}") 
    print('Automated task scheduling is working properly.')

@app.task
def automatic_ingest():
    print(">>> automatic ingest task triggered <<<")
    if not os.path.exists(f'{os.environ["TELEMETRY_ARCHIVE"]}/ingest.lock'):
        staged_files = get_staged_files()
        if staged_files:
            last_staged_h5_time = np.max([os.path.getctime(f) for f in staged_files])
            if Time.now().unix - last_staged_h5_time > 900:
                with open(f'{os.environ["TELEMETRY_ARCHIVE"]}/ingest.lock', 'w') as lock_file:
                    subprocess.run(
                        ['python',
                        '/srv/jeta/code/jeta/archive/ingest.py',
                        '>',
                        f'/srv/jeta/log/jeta.ingest.{Time.now().yday.replace(":", "").replace(".", "")}.log',
                        '2>&1',
                        '&']
                    )
                print('removing lock')
                os.remove(f'{os.environ["TELEMETRY_ARCHIVE"]}/ingest.lock')
            else:
                print('skipping ingest, files are still being uploaded to staging, nothing to do ... :)')
                return 0
        else:
            print('skipping ingest, staging is empty, nothing to do ... :)')
            return 0
    else:
        print('skipping ingest, ingest is already running, nothing to do ... :)')
        return 0
