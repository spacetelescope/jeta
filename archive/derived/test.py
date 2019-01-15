# Licensed under a 3-clause BSD style license - see LICENSE.rst
from . import base

class DerivedParameterTest(base.DerivedParameter):
    content_root = 'test'

#--------------------------------------------
class DP_TEST1(DerivedParameterTest):
    rootparams = ['TEPHIN']
    time_step = 32.8

    def calc(self, dataset):
        return dataset['TEPHIN'].vals

class DP_TEST2(DerivedParameterTest):
    rootparams = ['TEPHIN']
    time_step = 65.6

    def calc(self, dataset):
        return dataset['TEPHIN'].vals * 1.5

class DP_TEST3(DerivedParameterTest):
    rootparams = ['TEPHIN']
    time_step = 65.6

    def calc(self, dataset):
        return dataset['TEPHIN'].vals * 2.0

