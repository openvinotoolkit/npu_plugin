
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

import operator

from .Layer import Layer
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable


class Permute(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfIV = TensorFormat(Layouts.NHCW, (1, 2, 3))
        tfCM = TensorFormat(Layouts.NHWC, (1, 2, 3))
        tfPL = TensorFormat(Layouts.NCHW, (1, 2, 3))
        self.formatPool = [(tfIV, tfIV), (tfIV, tfPL), (tfIV, tfCM),
                           (tfCM, tfCM), (tfCM, tfPL), (tfCM, tfIV),
                           (tfPL, tfPL), (tfPL, tfCM), (tfPL, tfIV)]

    def load_order_from_caffe(self, caffe_order):
        # Caffe tensors can have less than 3 axes + batch axis. MvTensor supports
        # only 3 axes minus batch axis.
        if len(caffe_order) != 4:
            throw_error(
                ErrorTable.StageDetailsNotSupported,
                "Support only for 3 axes minus batch axis")

        # Permutation of the batch axis is not supported in MvTensor.
        if caffe_order[0] != 0:
            throw_error(
                ErrorTable.StageDetailsNotSupported,
                "Permutation of the batch axis is not supported")

        # MvTensor does not support batch axis and the axes are numbered
        # in increasing order from the slowest increasing axis.
        # Remove batch axis and subtract 1.
        caffe_order = caffe_order[1:]
        caffe_order = [x - 1 for x in caffe_order]
        caffe_perm = [-1, -1, -1]
        for i, p_i in enumerate(caffe_order):
            caffe_perm[p_i] = i

        assert(all([x >= 0 for x in caffe_perm]))

        self.permutation = caffe_perm
        self.reference_layout = Layouts.NCHW

    @staticmethod
    def compute_layout_permutation(from_layout, to_layout, inverse=False):
        """
        Computes and returns the list specifying the direct/inverse permutation
        from the from_layout to the to_layout.

        Keyword arguments:
        from_layout -- layout tuple
        to_layout -- layout tuple
        inverse -- if True returns the invers permutation (default False)

        return: permutation list or None for error.
        """

        if (len(from_layout) != 4 or len(to_layout) != 4):
            return None

        permutation = [-1, -1, -1, -1]
        for from_axis_i, from_axis in enumerate(from_layout):
            for to_axis_i, to_axis in enumerate(to_layout):
                if (from_axis == to_axis):
                    permutation[from_axis_i] = to_axis_i
                    break

        if (any([x < 0 for x in permutation])):
            return None

        if (not inverse):
            return permutation
        else:
            inv_permutation = [-1, -1, -1, -1]
            for i in range(0, len(permutation)):
                for axis_i, axis in enumerate(permutation):
                    if (axis == i):
                        inv_permutation[i] = axis_i

            return inv_permutation

    def get_mvtensor_permutation(self, input_layout, output_layout):
        """
        Returns the permutation of the axis to be executed on myriad.

        caffe input a0a1a2 ---Pcaffe---> caffe output
                      |                       |
                     Pin                     Pout
                      |                       |
        MvTensor    input   ----Pmvt--->  mvt output
        Pmvt = Pin^-1 * Pcaffe * Pout^-1
        """
        inv_perm_ref_to_in = self.compute_layout_permutation(
            self.reference_layout, input_layout, True)
        assert(inv_perm_ref_to_in is not None)
        inv_perm_out_to_ref = self.compute_layout_permutation(
            output_layout, self.reference_layout, True)
        assert(inv_perm_out_to_ref is not None)

        # MvTensor does not support batch axis and the axes are numerotated
        # in increasing order from the slowest increasing axis.
        # Remove batch axis and subtract 1.
        inv_perm_ref_to_in = [x - 1 for x in inv_perm_ref_to_in[1:]]
        inv_perm_out_to_ref = [x - 1 for x in inv_perm_out_to_ref[1:]]

        permutation = operator.itemgetter(
            inv_perm_ref_to_in[0],
            inv_perm_ref_to_in[1],
            inv_perm_ref_to_in[2])(
            self.permutation)

        permutation = operator.itemgetter(permutation[0],
                                          permutation[1],
                                          permutation[2])(inv_perm_out_to_ref)

        return permutation


class PermuteFlatten(Permute):
    def __init__(self, base):
        super().__init__(
            base.getStringifiedName(),
            base.inputTensorNames,
            base.outputTensorNames)

        # Copy the attributes
        for attr_name in base.__dict__:
            setattr(self, attr_name, getattr(base, attr_name))

    def get_mvtensor_permutation(self, input_layout, output_layout):
        """
        Returns the permutation of the axis to be executed on myriad.

        apply the Flatten's permutation on top of the Permute's permutation
        basically, Flatten is an identical permutation followed by a reshape
        to [a0*a1*a2, 1, 1]
        caffe input a0a1a2 ---Pcaffe---> caffe output
                      |                       |
                     Pin                     Pout
                      |                       |
        MvTensor    input   ----Pmvt--->  mvt output
        Pmvt = Pin^-1 * Pcaffe-perm * Pcaffe-flatten * Pout^-1
        However, since Pout^-1 is performed on the volume [a0*a1*a2, 1, 1], then
        Pout^-1 is also the identical permutation, so:
        Pmvt = Pin^-1 * Pcaffe-perm
        """

        inv_perm_ref_to_in = self.compute_layout_permutation(
            self.reference_layout, input_layout, True)
        assert(inv_perm_ref_to_in is not None)
        inv_perm_ref_to_in = [x - 1 for x in inv_perm_ref_to_in[1:]]
        permutation = operator.itemgetter(
            inv_perm_ref_to_in[0],
            inv_perm_ref_to_in[1],
            inv_perm_ref_to_in[2])(
            self.permutation)

        return permutation
