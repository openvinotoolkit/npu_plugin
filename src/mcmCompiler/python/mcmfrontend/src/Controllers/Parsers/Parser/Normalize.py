
# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

from .Layer import Layer
from Controllers.Tensor import PopulatedTensor
from Controllers.TensorFormat import TensorFormat
import Models.Layouts as layouts
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable


class Normalize(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        tfCM = TensorFormat(layouts.NHWC, concatAxes=(1, 2, 3))
        tfIV = TensorFormat(layouts.NHCW, concatAxes=(1, 2, 3))
        tfPL = TensorFormat(layouts.NCHW, concatAxes=(1, 2, 3))

        self.formatPool = [(tfIV, tfIV), (tfPL, tfPL),
                           (tfPL, tfIV), (tfIV, tfPL), (tfCM, tfCM)]

    def load_parameters(self, normalize_parameters):
        if(normalize_parameters.across_spatial is not False):
            throw_error(
                ErrorTable.StageDetailsNotSupported,
                "Only Values of False accepted for Across Spatial, Normalization")
        self.across_spatial = normalize_parameters.across_spatial

        if(normalize_parameters.channel_shared is not False):
            throw_error(
                ErrorTable.StageDetailsNotSupported,
                "Only Values of False accepted for Channel Shared, Normalization")

        self.channel_shared = normalize_parameters.channel_shared

        self.epsilon = normalize_parameters.eps

    def load_scales(self, scales):
        self.scales = PopulatedTensor(scales)
