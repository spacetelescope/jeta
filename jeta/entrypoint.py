# _*_ coding: utf-8 _*_
"""Update the telemetry engineering archive database.

This module is intended to be run automatically at regular a
regular interval.

"""

if __name__ == '__main__':
    from jeta.archive import update
    update.begin()
