from __future__ import absolute_import, unicode_literals

from celery import Celery
from celery.schedules import crontab

app = Celery(
    broker='redis://localhost',
    backend='redis://localhost',
)

app.conf.update(
    result_expires=3600,
)

app.autodiscover_tasks()

app.conf.beat_schedule = {
    'verify_scheduling_works': {
        'task': 'scheduling_check',
        'schedule': crontab(minute='*/1'),
    }
}

if __name__ == '__main__':
    app.start()