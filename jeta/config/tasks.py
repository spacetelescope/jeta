import datetime
from .celery import app

@app.task
def scheduling_check():
    print(f"The task was run {datetime.datetime.now()}")
