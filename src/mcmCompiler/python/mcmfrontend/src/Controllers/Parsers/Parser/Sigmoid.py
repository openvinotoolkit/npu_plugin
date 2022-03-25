#!/usr/bin/env python3
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from Controllers.TensorFormat import TensorFormat
import Models.Layouts as Layouts
from .Layer import Layer


class Sigmoid(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (2,))
        self.formatPool = [(tfCM, tfCM)]
