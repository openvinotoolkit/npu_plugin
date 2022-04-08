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


class AxisAlignment():
    def getAchievableValue(self, axisDim):
        raise NotImplementedError


class AxisConstantAlignment():
    """
        Round up the axis dimension to be a multiple of a constant
        (which is provided).
    """

    def __init__(self, constant):
        self.constant = constant
        assert(self.constant > 0)

    def __roundup(self, x, mult):
        return (x + mult - 1) // mult * mult

    def getAchievableValue(self, axisDim):
        return self.__roundup(axisDim, self.constant)


class AxisPowerAlignment():
    """
        Round up the axis dimension to much
    """
    pass


class TensorFormat():
    def __init__(
        self,
        layout,
        concatAxes,
        startAlignment=0,
        axesAlignment=(
            1,
            1,
            1,
            1)):
        """
            Defaults assume a 'tight' buffer that starts at the origin and
            each element has no specific alignment
        """
        self.layout = layout
        self.concatAxes = concatAxes
        self.startAlignment = startAlignment
        self.axesAlignment = axesAlignment

    def __roundup(self, x, mult):
        return (x + mult - 1) // mult * mult

    def compatible(self, tensorFormat, shape, tensorAxisDemand):
        """
            Performs several checks between this object and @tensorFormat.
            - Comparison of layouts
        """

        # Check whether layout matches
        if self.layout != tensorFormat.layout:
            return False

        # Loop though the axes and check
        for axisIdx in reversed(range(len(self.layout))):
            # Check primary axis alignment
            primaryAxis = self.layout[axisIdx]
            primaryDimension = shape[primaryAxis]
            primaryStrideA = self.__roundup(
                primaryDimension, self.axesAlignment[primaryAxis])
            primaryStrideB = self.__roundup(
                primaryDimension, tensorFormat.axesAlignment[primaryAxis])

            axisDemanded = bool(tensorAxisDemand[axisIdx])

            if axisDemanded and axisIdx not in self.concatAxes:
                return False

            if axisDemanded and axisIdx not in tensorFormat.concatAxes:
                return False

            if primaryStrideA != primaryStrideB and (
                    primaryAxis not in self.concatAxes or primaryAxis not in tensorFormat.concatAxes):
                return False

            # Check if concat axis exists to generate this stride
            if primaryStrideA % primaryDimension > 0 and primaryAxis not in self.concatAxes:
                return False

            if primaryStrideB % primaryDimension > 0 and primaryAxis not in tensorFormat.concatAxes:
                return False

        return True

    def ensureCompatibility(self, tensorFormat, shape):
        # Check whether layout matches
        assert(self.layout == tensorFormat.layout)

        # Just copy the shape of layout
        newShape = list(self.layout)

        # Loop though the axes and check
        for axisIdx in reversed(range(len(self.layout))):
            # Check primary axis alignment
            primaryAxis = self.layout[axisIdx]
            primaryDimension = shape[primaryAxis]
            primaryStrideA = self.__roundup(
                primaryDimension, self.axesAlignment[primaryAxis])
            primaryStrideB = self.__roundup(
                primaryDimension, tensorFormat.axesAlignment[primaryAxis])

            newShape[primaryAxis] = max(primaryStrideA, primaryStrideB)

        return tuple(newShape)

    def getSignature(self):
        return 'layout: {} - concatAxes: {} - startAlignment: {} - axesAlignment: {}'.format(
            self.layout, self.concatAxes, self.startAlignment, self.axesAlignment)
