#!/usr/bin/env python3

import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat
from .Layer import Layer


class Bias(Layer):

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]

    def loadTrainedParameters(self, **kwargs):
        pass

    def loadBias(self, bias):
        self.bias = bias

    def getBias(self):
        return self.bias
