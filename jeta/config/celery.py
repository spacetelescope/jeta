from __future__ import absolute_import, unicode_literals

from celery import Celery
from celery.schedules import crontab

from . import celeryconfig

app = Celery(
    broker='redis://localhost',
    backend='redis://localhost',
)

app.config_from_object(celeryconfig)

app.conf.update(
    result_expires=3600,
)

app.autodiscover_tasks()

app.conf.beat_schedule = {
    'verify_scheduling_works': {
        'task': 'jeta.config.tasks.debug_task',
        'schedule': crontab(minute='*/30'),
        'args': ()
    },
    'automatic_ingest': {
        'task': 'jeta.config.tasks.automatic_ingest',
        'schedule': crontab(minute='*/2'),
        'args': ()
    },
}

if __name__ == '__main__':
    print('Called from CLI, nothing do to.')
