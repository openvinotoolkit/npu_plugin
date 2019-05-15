# Copyright 2017 Intel Corporation.
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

import os
import subprocess
import tempfile
import sys
import random
import math
import re

# Tensorflow 1.7 has some deprecated features that result in warnings when it is
# imported. Suppress these warnings until TF resolves them.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

import google.protobuf as proto
import numpy as np
import networkx as nx

from collections import OrderedDict
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops

from parserTools.EnumController import throw_error, ErrorTable, compiler_assert
from parserTools.Parser.Layer import OriginalName, MangledName
from parserTools.Parser.Output import Output
from parserTools.Parser.Input import Input
from parserTools.Parser.DetectionOutput import DetectionOutput
from parserTools.Parser.Bias import Bias
from parserTools.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
from parserTools.Quantization import QuantizationParameters

from parserTools import TensorFlowLiteParser as tfp
from parserTools.TensorFlowLiteParser.Helpers import getTensorType
from parserTools.TensorFlowLiteParser.tflite.Model import Model
from parserTools.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from parserTools.Tensor import UnpopulatedTensor

from parserTools.MiscIO import parse_img, preprocess_img
from parserTools.GraphUtils import buildGraph, buildLayerLists

from ordered_set import OrderedSet

def regularizeInPlaceOps(parsedLayers):
    # Some operations in Caffe can be inplace. Introduce new names for these layers
    inPlaceOps = OrderedDict()
    tensorProducer = OrderedDict()
    for layer in parsedLayers:
        for i in layer.getOutputTensorNames():
            try:
                tensorProducer[i.stringifyName()].append(layer)
            except:
                tensorProducer[i.stringifyName()] = [layer]

        if OrderedSet(layer.getOutputTensorNames()).intersection(OrderedSet(layer.getInputTensorNames())):
            assert(len(layer.getOutputTensorNames()) == 1)

            tensorName = layer.getOutputTensorNames()[0]
            try:
                inPlaceOps[tensorName.stringifyName()].append(layer)
            except:
                inPlaceOps[tensorName.stringifyName()] = [layer]

    def remangleName(mangledNameList, matchingName):
        for idx, n in enumerate(mangledNameList):
            if n.stringifyName() == matchingName:
                newName = n.remangle()
                mangledNameList[idx] = newName
                return newName

    def replaceName(mangledNameList, matchingName, newName):
        for idx, n in enumerate(mangledNameList):
            if n.stringifyName() == matchingName:
                mangledNameList[idx] = newName

    for tensorName, layerGroup in inPlaceOps.items():
        extendedList = list(OrderedSet(tensorProducer[tensorName]).difference(OrderedSet(layerGroup)))
        extendedList.extend(layerGroup[:-1])
        for producer, consumer in zip(extendedList, layerGroup):
            newName = remangleName(producer.getOutputTensorNames(), tensorName)
            replaceName(consumer.getInputTensorNames(), tensorName, newName)

    return parsedLayers

def createTensors(parsedLayers):
    # Collect all the tensorNames and sizes
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensorName, tensorSize in zip(layer.getInputTensorNames(), layer.getInputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName()] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

        for tensorName, tensorSize in zip(layer.getOutputTensorNames(), layer.getOutputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName()] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

    for layer in parsedLayers:
        layer.setInputTensors([tensorNames[n.stringifyName()] for n in layer.getInputTensorNames()])
        layer.setOutputTensors([tensorNames[n.stringifyName()] for n in layer.getOutputTensorNames()])

def quantizeTensors(parsedLayers, tensors):
    # Only for UnpopulatedTensors, populated dtype is taken from data
    tensors_list = list(OrderedSet(sum([list(l.getInputTensors()) + list(l.getOutputTensors()) for l in parsedLayers], [])))
    tensor_quantization_map = {str(tensors[idx].Name().decode("utf-8")):tensors[idx].Quantization() for idx in range(len(tensors))}
    tensor_name_map = {str(tensors[idx].Name().decode("utf-8")):tensors[idx] for idx in range(len(tensors))}

    for t in tensors_list:
        tName = t.name.stringifyOriginalName()
        dt = getTensorType(tensor_name_map[tName])
        if dt == np.float32:
            dt = np.float16
        t.setDatatype(dt)
        q = tensor_quantization_map[tName]
        if q:
            tensor_range = [(q.Min(idx), q.Max(idx)) for idx in range(q.MinLength())]
            t.setQuantizationParameters(QuantizationParameters(q.ScaleAsNumpy(), q.ZeroPointAsNumpy(), dt, tensor_range))


def insertInputOutputOps(parsedLayers, input_name, output_name):
    # Find all tensors that are not consumed by anybody
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensor in layer.getInputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([0, 0], tensor, [])

            tensorNames[tensor.getName().stringifyName()][0][0] += 1

        for tensor in layer.getOutputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([0, 0], tensor, [])

            tensorNames[tensor.getName().stringifyName()][0][1] += 1
            tensorNames[tensor.getName().stringifyName()][2].append(layer)

    for tensorName, tensorValue in tensorNames.items():
        consumersAndProducers, tensor, producers = tensorValue
        consumers = consumersAndProducers[0]
        producers_cnt = consumersAndProducers[1]
        if consumers == 0 and tensor.getName().stringifyOriginalName() == output_name:
            x = Output('output', [tensor.getName()], [])
            x.setInputTensors([tensor])
            x.setOutputTensors([])
            x.loadInputTensorSizes([tensor.getShape()])
            x.loadOutputTensorSizes([])

            assert(len(producers) == 1)
            if isinstance(producers[0], DetectionOutput):
                x.enableDetectionOutput()

            parsedLayers.append(x)
        if producers_cnt == 0 and tensor.getName().stringifyOriginalName() == input_name:
            x = Input('input', [], [])
            x.setOutputTensorsAllFields([tensor])
            x.setInputTensorsAllFields([])

            parsedLayers.append(x)

    return parsedLayers

def printLayers(layers):
    for layer in layers:
        print('Node:', layer)
        print('  layer name:', layer.getStringifiedName())
        print('  input tensors:', )
        for tensorName in layer.getInputTensorNames():
            print('    ', tensorName.stringifyName())
        print('  output tensors:')
        for tensorName in layer.getOutputTensorNames():
            print('    ', tensorName.stringifyName())


class TensorFlowLiteParser:

    def __init__(self):
        self.type = 'TensorFlowLite'
        self.model = None
        self.model_path = None
        self.feed_dict = None

    def getType(self):
        return self.type

    # calculate the reference output of the graph to compare with myriad results
    def calculateReference(self, arguments):

        image = arguments.image
        seed = 1
        raw_scale = 1
        mean = None
        channel_swap = None
        input_node_name = None
        output_node_name = None

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.contrib.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(input_details) != 1:
            raise ValueError("Only a single input is supported (instead found {}".format(len(input_details)))

        if len(output_details) != 1:
            raise ValueError("Only a single output is supported (instead found {}".format(len(output_details)))
        input_details = input_details[0]
        output_details = output_details[0]
        shape = input_details['shape']

        if image is None or image == "Debug":

            if seed != -1:
                np.random.seed(seed)
                random.seed(seed)
            np.random.seed(0)
            random.seed(0)
            input_data = np.random.uniform(0, 256, shape).astype(input_details['dtype'])
            print("Input image shape", shape)
            input_data = preprocess_img(input_data,
                                        raw_scale=raw_scale,
                                        mean=mean,
                                        channel_swap=channel_swap)
        else:
            input_data = parse_img(image,
                                [int(shape[0]),
                                    int(shape[3]),
                                    int(shape[1]),
                                    int(shape[2])],
                                raw_scale=raw_scale,
                                mean=mean,
                                channel_swap=channel_swap,
                                dtype=input_details['dtype'])
            input_data = input_data.transpose([0, 2, 3, 1])

        self.interpreter.set_tensor(input_details['index'], input_data)

        self.interpreter.invoke()
        expected_result = self.interpreter.get_tensor(output_details['index'])

        if input_node_name is not None:
            if not input_node_name in self.tensor_index_map.keys():
                throw_error(ErrorTable.GraphConstructionFailure, input_node_name)
            input_data = self.get_layer_data(arguments.input_node_name).transpose((0, 2, 3, 1))
        if output_node_name is not None:
            if not output_node_name in self.tensor_index_map.keys():
                throw_error(ErrorTable.GraphConstructionFailure, output_node_name)
            expected_result = self.get_layer_data(output_node_name).transpose((0, 2, 3, 1))

        # convert shape
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        if len(expected_result.shape) == 4:
            expected_result = np.transpose(expected_result, (0, 3, 1, 2))
        elif len(expected_result.shape) == 3:
            pass
        elif len(expected_result.shape) == 2:
            expected_result = expected_result.reshape(1, expected_result.shape[1], expected_result.shape[0])
        else:
            expected_result = expected_result.reshape(1, 1, expected_result.shape[0])

        np.save("Fathom_expected.npy", expected_result)

        return input_data, expected_result, output_details['name']


    def loadNetworkObjects(self,graph_path, model_path=None):
        """ Get the tensorflow protobuff model and parse it via tensorflow
        """
        print(graph_path)
        with open(graph_path, "rb") as fp:
            buf = fp.read()

        self.model = Model.GetRootAsModel(buf, 0)
        self.model_path = graph_path

        return

    def parse(self):

        # Define which subparser needs to be called for each layer
        su = {
            BuiltinOperator.AVERAGE_POOL_2D : tfp.Pooling.load,
            BuiltinOperator.ADD : tfp.Eltwise.load,
            BuiltinOperator.CONCATENATION : tfp.Concat.load,
            BuiltinOperator.CONV_2D : tfp.Convolution.load,
            BuiltinOperator.FULLY_CONNECTED : tfp.MatMul.load,
            BuiltinOperator.DEPTHWISE_CONV_2D : tfp.Convolution.load,
            BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION : tfp.LRN.load,
            BuiltinOperator.MAX_POOL_2D : tfp.Pooling.load,
            BuiltinOperator.MUL : tfp.Eltwise.load,
            BuiltinOperator.RELU : tfp.ReLU.load,
            BuiltinOperator.RELU6 : tfp.ReLU.load,
            BuiltinOperator.LEAKY_RELU: tfp.ReLU.load,
            BuiltinOperator.PRELU : tfp.ReLU.load,
            BuiltinOperator.SOFTMAX : tfp.Softmax.load,
            BuiltinOperator.TANH : tfp.Tanh.load,
            BuiltinOperator.LOGISTIC : tfp.Sigmoid.load,
            BuiltinOperator.PAD : tfp.Pad.load,
            BuiltinOperator.SUB : tfp.Eltwise.load,
            BuiltinOperator.RESHAPE : tfp.NoOp.load,
            # BuiltinOperator.DIV : None,
            # BuiltinOperator.SLICE : None,
            BuiltinOperator.SUM : tfp.Eltwise.load,
            BuiltinOperator.TRANSPOSE_CONV : tfp.Convolution.load,
            # BuiltinOperator.SQRT : None
        }

        parsedLayers = []
        version = self.model.Version()
        description = self.model.Description()
        buffers = [self.model.Buffers(idx) for idx in range(self.model.BuffersLength())]
        subgraphs = [self.model.Subgraphs(idx) for idx in range(self.model.SubgraphsLength())]
        operator_codes = [self.model.OperatorCodes(idx) for idx in range(self.model.OperatorCodesLength())]

        if len(subgraphs) != 1:
            raise ValueError("Only a single subgraph is supported in the current TFLite")

        sg = subgraphs[0]
        tensors = [sg.Tensors(idx) for idx in range(sg.TensorsLength())]
        input_idxs = [sg.Inputs(idx) for idx in range(sg.InputsLength())]
        output_idxs = [sg.Outputs(idx) for idx in range(sg.OutputsLength())]
        operators = [sg.Operators(idx) for idx in range(sg.OperatorsLength())]

        self.tensor_index_map = {str(tensors[idx].Name().decode("utf-8")):idx for idx in range(len(tensors))}

        if len(output_idxs) != 1:
            raise ValueError("Unsupported more than one output")
        if len(input_idxs) != 1:
            raise ValueError("Unsupported more than one input")

        for obj in operators:
            op_type = operator_codes[obj.OpcodeIndex()].BuiltinCode()
            opParser = su.get(op_type, None)

            compiler_assert(opParser is not None,
                        ErrorTable.StageTypeNotSupported, op_type)
            if opParser is not None:
                parsedLayers.extend(opParser(obj, op_type, tensors, buffers))

        # Mangle tensor names
        tensorNamesDict = OrderedDict()
        for layer in parsedLayers:
            for tensorName in list(layer.getInputTensorNames()) + list(layer.getOutputTensorNames()):
                if tensorName not in tensorNamesDict:
                    tensorNamesDict[tensorName] = MangledName(OriginalName(tensorName))

        # Replace tensor names in layers with mangled ones:
        for layer in parsedLayers:
            layer.setInputTensorNames([tensorNamesDict[name] for name in layer.getInputTensorNames()])
            layer.setOutputTensorNames([tensorNamesDict[name] for name in layer.getOutputTensorNames()])

        # Convert inPlace operations into regular ones
        parsedLayers = regularizeInPlaceOps(parsedLayers)

        # Create tensor objects for each operation
        createTensors(parsedLayers)

        # Add quantization fields to quantized tensors
        quantizeTensors(parsedLayers, tensors)

        # Introduce Output operations
        output_tensor_name = str(tensors[output_idxs[0]].Name().decode("utf-8"))
        input_tensor_name = str(tensors[input_idxs[0]].Name().decode("utf-8"))
        parsedLayers = insertInputOutputOps(parsedLayers, input_tensor_name, output_tensor_name)

        print("Fusing Add and Batch after Convolution")
        g = buildGraph(parsedLayers)
        parsedLayers = buildLayerLists(g)

        return parsedLayers

    def get_layer_data(self, outBlobName):

        outBlobName = str(outBlobName)

        if not outBlobName in self.tensor_index_map.keys():
            raise ValueError("Impossible to get tensor {}".format(outBlobName))

        out_data = self.interpreter.get_tensor(self.tensor_index_map[outBlobName])

        if (len(out_data.shape) == 4):
            out_data = out_data.transpose((0, 3, 1, 2))
        return out_data



# def fuseBiasAdd(g):
#     from parserTools.Parser.InnerProduct import InnerProduct
#     from parserTools.Optimizer import fuse_nodes

#     """
#         Iterates over the graph removing any qualifying fusions for
#         bias and add until we are complete.
#     """

#     def isBiasOrAdd(layer):
#         """
#             Returns True/False if the given layer is/is not a Bias or Add Layer
#         """
#         from parserTools.Parser.Bias import Bias
#         from parserTools.Parser.Eltwise import Eltwise
#         return (type(layer) == Bias) or \
#                 ((type(layer) == Eltwise) and (layer.getType() == Eltwise.Type.WSUM))

#     def isConvOrFC(layer):
#         """
#             Returns True/False if the given layer is/is not a Convolution or InnerProduct Layer
#         """
#         from parserTools.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
#         return type(layer) in [Convolution2D, ConvolutionDepthWise2D, InnerProduct]

#     def PutBiasInConv(layer, absorbed_layer):
#         """
#             To account for the removal of Bias and Add, we insert the bias
#         """
#         from parserTools.Parser.Bias import Bias
#         from parserTools.Parser.Eltwise import Eltwise

#         if (type(absorbed_layer) == Bias):
#             b = absorbed_layer.getBias()
#         else:
#             # TODO: pull the data out of the Add input
#             b = 0
#             assert("TensorFlow: Add after Matmul is not supported")

#         # Change Bias
#         if layer.biasEnabled():
#             if b is not None:
#                 layer.setBias(layer.getBias().data + b)
#         else:
#             # If there is no bias, it is possible that we will need to now have one
#             if b is not None:
#                 layer.setBiasEnabled(True)
#                 layer.setBias(np.array(b).astype(np.float16))
#         return layer

#     check_again = True
#     while check_again:
#         g, check_again = fuse_nodes(g, isBiasOrAdd, isConvOrFC, PutBiasInConv)

#     return g