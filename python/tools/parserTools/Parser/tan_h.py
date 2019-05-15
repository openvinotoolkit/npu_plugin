#!/usr/bin/env python3

from .Layer import Layer
import parserTools.Models.Layouts as Layouts
from parserTools.TensorFormat import TensorFormat


class TanH(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]