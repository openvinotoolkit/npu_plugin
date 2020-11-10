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

import Models.Layouts as Layouts
from copy import copy

mangledNameCounter = 0


def getAndIncreaseManglingCounter():
    global mangledNameCounter
    mangledNameCounter += 1
    return mangledNameCounter - 1


class OriginalName():
    def __init__(self, name):
        if isinstance(name, bytes):
            self.name = name.decode("utf-8")
        else:
            self.name = name

    def getName(self):
        return self.name


class MangledName():
    def __init__(self, origName):
        self.origName = origName
        self.mangleValue = getAndIncreaseManglingCounter()

    def remangle(self):
        return MangledName(self.origName)

    def __str__(self):
        return self.stringifyName()

    def stringifyName(self):
        return self.origName.getName() + '#{}'.format(self.mangleValue)

    def stringifyOriginalName(self):
        return self.origName.getName()


class Layer():
    def __init__(self, name, inputTensorNames, outputTensorNames):
        self.name = MangledName(OriginalName(name))
        self.inputTensorNames = inputTensorNames
        self.outputTensorNames = outputTensorNames
        self.isHW = False
        self.compatible_layouts = [Layouts.NHWC]  # Channel Minor defaut
        self.implicit = False
        self.inPlace = False
        self.sliced = False

    def __str__(self):
        return str(self.__class__.__name__) + ":" + str(self.name)

    def __deepcopy__(self, memo):
        """
            NetworkX performs a deep copy in several operations (e.g. edge contraction).
            We do not want this, this we may end up having multiple copies of an object.
        """
        return copy(self)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def className(self):
        return self.__class__.__name__.lower()

    def getStringifiedName(self):
        return self.name.stringifyName()

    def getStringifiedOriginalName(self):
        return self.name.origName.name

    def getInputTensorsCount(self):
        return len(self.inputTensorNames)

    def getInputTensorNames(self):
        return self.inputTensorNames

    def setInputTensorNames(self, inputTensorNames):
        self.inputTensorNames = list(inputTensorNames)

    def setInputTensors(self, t):
        self.inputTensors = tuple(t)

    def getInputTensors(self):
        return self.inputTensors

    def getOutputTensorsCount(self):
        return len(self.outputTensorNames)

    def getOutputTensorNames(self):
        return self.outputTensorNames

    def setOutputTensorNames(self, outputTensorNames):
        self.outputTensorNames = list(outputTensorNames)

    def setOutputTensors(self, t):
        self.outputTensors = tuple(t)

    def getOutputTensors(self):
        return self.outputTensors

    def setInputTensorsAllFields(self, tensors):
        self.inputTensors = ()
        self.inputTensorNames = []
        self.inputTensorSizes = []
        for t in tensors:
            self.inputTensors = self.inputTensors + tuple([t])
            self.inputTensorNames.append(t.getName())
            self.inputTensorSizes.append(t.shape)

    def appendInputTensorsAllFields(self, tensors):
        for t in tensors:
            self.inputTensors = self.inputTensors + tuple([t])
            self.inputTensorNames.append(t.getName())
            self.inputTensorSizes.append(t.shape)

    def removeInputTensorsAllFields(self, tensors):
        for t in tensors:
            try:
                self.inputTensors = tuple(
                    x for x in self.inputTensors if x.name.stringifyName() != t.name.stringifyName())
                self.inputTensorNames.remove(t.getName())
                self.inputTensorSizes.remove(t.shape)
            except ValueError:
                print("Warning: Input Tensor requested for removal, but not present.")

    def setOutputTensorsAllFields(self, tensors):
        self.outputTensors = ()
        self.outputTensorNames = []
        self.outputTensorSizes = []
        for t in tensors:
            self.outputTensors = self.outputTensors + tuple([t])
            self.outputTensorNames.append(t.getName())
            self.outputTensorSizes.append(t.shape)

    def appendOutputTensorsAllFields(self, tensors):
        for t in tensors:
            self.outputTensors = self.outputTensors + tuple([t])
            self.outputTensorNames.append(t.getName())
            self.outputTensorSizes.append(t.shape)

    def loadTensorSizes(self, tensorDict):
        self.inputTensorSizes = []
        for name in self.inputTensorNames:
            self.inputTensorSizes.append(tensorDict[name])

        self.outputTensorSizes = []
        for name in self.outputTensorNames:
            self.outputTensorSizes.append(tensorDict[name])

    def loadInputTensorSizes(self, sizes):
        self.inputTensorSizes = list(sizes)

    def loadOutputTensorSizes(self, sizes):
        self.outputTensorSizes = list(sizes)

    def getInputTensorSizes(self):
        return tuple(self.inputTensorSizes)

    def getOutputTensorSizes(self):
        return tuple(self.outputTensorSizes)

    def loadTrainedParameters(self, **kwargs):
        pass

    def setImplicit(self):
        """
            This layer is structurally significant, but in practise is not
            computed.
        """
        self.implicit = True

    def unsetImplicit(self):
        """
            Undo the 'setImplicit' function
        """
        self.implicit = False

    def getImplicit(self):
        """
            Returns False if layer is a nessicary compute step
            Else: True
        """
        return self.implicit

    def getCompatibleLayouts(self):
        """
            Returns layouts supported by this layer.
            Default is [Channel Minor]
        """
        return self.compatible_layouts

    def addCompatibleLayout(self, layout):
        """
            Adds a layout to the list of supported ones.
            Only to be called in the initialization function of a layer type.
        """
        self.compatible_layouts.append(layout)

    def setLayoutIndependent(self):
        """
            Some layer only care that there is data, not what format.
            This function makes sure that all formats are set for the layer's
            compatiblity parameters
        """
        self.compatible_layouts = [
            Layouts.NCHW,
            Layouts.NHCW,
            Layouts.NHWC,
            (0, 1, 3, 2),
            (0, 3, 2, 1),
            (0, 3, 1, 2),
        ]

    def setInPlace(self):
        self.inPlace = True

    def getInPlace(self):
        return self.inPlace

    def forceLayout(self, layout_string):

        from Controllers.TensorFormat import TensorFormat

        layout = None
        if layout_string == "CHW":
            layout = Layouts.NCHW
        if layout_string == "HCW":
            layout = Layouts.NHCW
        if layout_string == "HWC":
            layout = Layouts.NHWC

        assert layout is not None, "Invalid Software Layout."
        tf = TensorFormat(layout, (1, 2, 3))

        self.formatPool = [(tf, tf)]
