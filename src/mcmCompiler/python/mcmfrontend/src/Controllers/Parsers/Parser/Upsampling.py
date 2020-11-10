#!/usr/bin/env python3

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


from Controllers.TensorFormat import TensorFormat
import Models.Layouts as Layouts
from .Layer import Layer


class Upsampling(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        tfCM = TensorFormat(Layouts.NHWC, (0, 1, 2, 3))
        tfIV = TensorFormat(Layouts.NHCW, (0, 1, 2, 3))
        tfPL = TensorFormat(Layouts.NCHW, (0, 1, 2, 3))
        self.formatPool = [(tfIV, tfIV), (tfPL, tfPL), (tfCM, tfCM)]

        upsampling_factor_C = None
        upsampling_factor_W = None
        upsampling_factor_H = None

        output_pad_C = None
        output_pad_W = None
        output_pad_H = None

    def set_upsampling_factor(self, up_factor_C, up_factor_W, up_factor_H):
        assert(up_factor_C >= 1)
        assert(up_factor_W >= 1)
        assert(up_factor_H >= 1)

        self.upsampling_factor_C = up_factor_C
        self.upsampling_factor_W = up_factor_W
        self.upsampling_factor_H = up_factor_H

    def set_output_pad_size(self, pad_C, pad_W, pad_H):
        assert(len(pad_C) >= 1 and len(pad_C) <= 2)
        assert(len(pad_W) >= 1 and len(pad_W) <= 2)
        assert(len(pad_H) >= 1 and len(pad_H) <= 2)

        self.output_pad_C = pad_C if len(pad_C) == 2 else [pad_C[0], pad_C[0]]
        self.output_pad_W = pad_W if len(pad_W) == 2 else [pad_W[0], pad_W[0]]
        self.output_pad_H = pad_H if len(pad_H) == 2 else [pad_H[0], pad_H[0]]

        assert(all([pad >= 0 for pad in self.output_pad_C]))
        assert(all([pad >= 0 for pad in self.output_pad_W]))
        assert(all([pad >= 0 for pad in self.output_pad_H]))

    def compute_and_set_upsampling_parameters_for_deconv(
            self,
            width_stride, height_stride,
            kernel_width, kernel_height,
            width_pad, height_pad
    ):
        """
        Helper function for seting the upsampling paramerters based on deconvolution
        parameters such that volume resulted from applying a Convolution with
        kernel of size kernel_width * kernel_height on the upsampled and padded
        volume is equal to the volume resulted from applying a Deconvolution on the
        original volume.
        """
        assert(width_stride >= 1)
        assert(height_stride >= 1)
        assert(kernel_width >= 1)
        assert(kernel_height >= 1)
        assert(width_pad >= 0)
        assert(height_pad >= 0)
        assert(len(self.inputTensorSizes) == 1)
        input_batch, input_channels, input_width, input_height = self.inputTensorSizes[0]

        self.set_upsampling_factor(1, width_stride, height_stride)
        pad_C = [0]
        pad_W = [kernel_width - width_pad - 1]
        pad_H = [kernel_height - height_pad - 1]

        self.set_output_pad_size(pad_C, pad_W, pad_H)

        self.__compute_output_shape()

    def __compute_output_shape(self):
        assert(len(self.inputTensorSizes) == 1)

        in_batch, in_channels, in_width, in_height = self.inputTensorSizes[0]

        out_batch = in_batch
        out_channels = self.output_pad_C[0] + self.output_pad_C[1] + \
            in_channels + (in_channels - 1) * (self.upsampling_factor_C - 1)
        out_width = self.output_pad_W[0] + self.output_pad_W[1] + in_width + \
            (in_width - 1) * (self.upsampling_factor_W - 1)
        out_height = self.output_pad_H[0] + self.output_pad_H[1] + in_height + \
            (in_height - 1) * (self.upsampling_factor_H - 1)

        self.outputTensorSizes = [
            [out_batch, out_channels, out_height, out_width]]
