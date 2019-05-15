#!/usr/bin/env python3

import parserTools.Models.Layouts as Layouts
from parserTools.TensorFormat import TensorFormat
from .Layer import Layer

class Sigmoid(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (2,))
        self.formatPool = [(tfCM, tfCM)]

