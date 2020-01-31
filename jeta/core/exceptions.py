"""
This module is used to define custom exceptions. The intent is to create
exceptions that are more descriptive than the defaults.
"""


class ImproperlyConfigured(Exception):

    def __init__(self, message):
        super().__init__(message)


class StrategyNotImplemented(Exception):

    def __init__(self, message):
        super().__init__(message)
