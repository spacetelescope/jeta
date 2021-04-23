from __future__ import absolute_import, unicode_literals

from celery import Celery
from celery.schedules import crontab

from jeta.archive.ingest import force_sort_by_time, calculate_delta_times

##CELERYBEAT_SCHEDULER = 'redbeat.RedBeatScheduler'

app = Celery(
    'jeta.archive',
    broker='redis://localhost',
    backend='redis://localhost',
    include=[
        'jeta.archive.ingest',
        'jeta.archive.ingest'
    ]
)

app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()


# app.conf.broker_url = 'redis://'
# app.conf.result_backend = 'redis://localhost'

# redbeat_redis_url = "redis://localhost"

# app.conf.beat_schedule = {
#     'execute_telemetry_ingest': {
#         'task': 'jeta.ingest.controller._execute_automated_ingest',
#         'schedule': crontab(minute='*/15'),
#     }
# }



# app.autodiscover_tasks()
