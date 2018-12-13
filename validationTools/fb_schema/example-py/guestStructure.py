import unittest
import os
import sys


class GuestRepr():
    def __init__(self):
        pass


class GuestLayer():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)