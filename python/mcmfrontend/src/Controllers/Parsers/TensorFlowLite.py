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

import Controllers.Parsers.Parser as parser
from Controllers.Parsers.BaseParser import BaseParser
from ordered_set import OrderedSet
from Controllers.GraphUtils import buildGraph, buildLayerLists
#from Controllers.MiscIO import parse_img, preprocess_img
from Controllers.Tensor import UnpopulatedTensor
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator
from Controllers.Parsers.TensorFlowLiteParser.tflite.Model import Model
from Controllers.Parsers.TensorFlowLiteParser.Helpers import getTensorType
from Controllers.Parsers import TensorFlowLiteParser as tfp
#from Controllers.InputToNetwork import parse_input
from Controllers.Quantization import QuantizationParameters
#from Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
#from Controllers.Parsers.Parser.Bias import Bias
from Controllers.Parsers.Parser.DetectionOutput import DetectionOutput
from Controllers.Parsers.Parser.Input import Input
from Controllers.Parsers.Parser.Output import Output
from Controllers.Parsers.Parser.Layer import OriginalName, MangledName
from Controllers.EnumController import throw_error, ErrorTable, compiler_assert
from collections import OrderedDict
import numpy as np
import random
import os

try:
    # lite module no longer in tf.contrib tf 1.14.0
    import tensorflow.contrib.lite as lite
except BaseException:
    import tensorflow.lite as lite


def regularizeInPlaceOps(parsedLayers):
    # Some operations in Caffe can be inplace. Introduce new names for these
    # layers
    inPlaceOps = OrderedDict()
    tensorProducer = OrderedDict()
    for layer in parsedLayers:
        for i in layer.getOutputTensorNames():
            try:
                tensorProducer[i.stringifyName()].append(layer)
            except BaseException:
                tensorProducer[i.stringifyName()] = [layer]

        if OrderedSet(
            layer.getOutputTensorNames()).intersection(
            OrderedSet(
                layer.getInputTensorNames())):
            assert(len(layer.getOutputTensorNames()) == 1)

            tensorName = layer.getOutputTensorNames()[0]
            try:
                inPlaceOps[tensorName.stringifyName()].append(layer)
            except BaseException:
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
        extendedList = list(
            OrderedSet(
                tensorProducer[tensorName]).difference(
                OrderedSet(layerGroup)))
        extendedList.extend(layerGroup[:-1])
        for producer, consumer in zip(extendedList, layerGroup):
            newName = remangleName(producer.getOutputTensorNames(), tensorName)
            replaceName(consumer.getInputTensorNames(), tensorName, newName)

    return parsedLayers


def createTensors(parsedLayers):
    # Collect all the tensorNames and sizes
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensorName, tensorSize in zip(
                layer.getInputTensorNames(), layer.getInputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName(
                )] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

        for tensorName, tensorSize in zip(
                layer.getOutputTensorNames(), layer.getOutputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName(
                )] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

    for layer in parsedLayers:
        layer.setInputTensors([tensorNames[n.stringifyName()]
                               for n in layer.getInputTensorNames()])
        layer.setOutputTensors([tensorNames[n.stringifyName()]
                                for n in layer.getOutputTensorNames()])


def quantizeTensors(parsedLayers, tensors):
    # Only for UnpopulatedTensors, populated dtype is taken from data
    tensors_list = list(OrderedSet(sum(
        [list(l.getInputTensors()) + list(l.getOutputTensors()) for l in parsedLayers], [])))
    tensor_quantization_map = {str(tensors[idx].Name().decode(
        "utf-8")): tensors[idx].Quantization() for idx in range(len(tensors))}
    tensor_name_map = {str(tensors[idx].Name().decode(
        "utf-8")): tensors[idx] for idx in range(len(tensors))}

    for t in tensors_list:
        tName = t.name.stringifyOriginalName()
        dt = getTensorType(tensor_name_map[tName])
        if dt == np.float32:
            dt = np.float16
        t.setDatatype(dt)
        q = tensor_quantization_map[tName]
        if q:
            tensor_range = [(q.Min(idx), q.Max(idx))
                            for idx in range(q.MinLength())]
            t.setQuantizationParameters(
                QuantizationParameters(
                    q.ScaleAsNumpy(),
                    q.ZeroPointAsNumpy(),
                    dt,
                    tensor_range))


def insertInputOutputOps(parsedLayers, input_name, output_name):
    # Find all tensors that are not consumed by anybody
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensor in layer.getInputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([
                    0, 0], tensor, [])

            tensorNames[tensor.getName().stringifyName()][0][0] += 1

        for tensor in layer.getOutputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([
                    0, 0], tensor, [])

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


class TensorFlowLiteParser(BaseParser):

    def __init__(self):
        self.type = 'TensorFlowLite'
        self.model = None
        self.model_path = None
        self.feed_dict = None

    def getType(self):
        return self.type

    # get a random input
    def getInput(self, arguments):

        # Load TFLite model and allocate tensors.
        self.interpreter = lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()

        if len(input_details) != 1:
            raise ValueError(
                "Only a single input is supported (instead found {}".format(
                    len(input_details)))
        input_details = input_details[0]
        shape = input_details['shape']
        dtype = input_details['dtype']
    #fixed seed for repeat inputs. 
        np.random.seed(0)
        random.seed(0)
        range_min = 0
        range_max = 256
        input_data = np.random.uniform(
            range_min, range_max, shape).astype(dtype)
    #hard-coded the path.:( 
        fp = open("./output/input.dat", 'wb')
        fp.write((input_data.flatten()).astype(np.uint8).data)
        fp.close()
        # convert shape
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return input_data

    def loadNetworkObjects(self, graph_path, model_path=None):
        """ Get the tensorflow protobuff model and parse it via tensorflow
        """
        print(graph_path)
        with open(graph_path, "rb") as fp:
            buf = fp.read()

        self.model = Model.GetRootAsModel(buf, 0)
        self.model_path = graph_path

        return

    def parse(self, arguments):

        # Define which subparser needs to be called for each layer
        subParsers = {
            BuiltinOperator.AVERAGE_POOL_2D: tfp.Pooling.load,
            BuiltinOperator.ADD: tfp.Eltwise.load,
            BuiltinOperator.CONCATENATION: tfp.Concat.load,
            BuiltinOperator.CONV_2D: tfp.Convolution.load,
            BuiltinOperator.FULLY_CONNECTED: tfp.MatMul.load,
            BuiltinOperator.DEPTHWISE_CONV_2D: tfp.Convolution.load,
            BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION: tfp.LRN.load,
            BuiltinOperator.MAX_POOL_2D: tfp.Pooling.load,
            BuiltinOperator.MUL: tfp.Eltwise.load,
            BuiltinOperator.RELU: tfp.ReLU.load,
            BuiltinOperator.RELU6: tfp.ReLU.load,
            BuiltinOperator.LEAKY_RELU: tfp.ReLU.load,
            BuiltinOperator.PRELU: tfp.ReLU.load,
            BuiltinOperator.SOFTMAX: tfp.Softmax.load,
            BuiltinOperator.TANH: tfp.Tanh.load,
            BuiltinOperator.LOGISTIC: tfp.Logistic.load,
            BuiltinOperator.PAD: tfp.Pad.load,
            BuiltinOperator.SUB: tfp.Eltwise.load,
            BuiltinOperator.RESHAPE: tfp.NoOp.load,
            # BuiltinOperator.DIV : None,
            # BuiltinOperator.SLICE : None,
            BuiltinOperator.SUM: tfp.Eltwise.load,
            BuiltinOperator.TRANSPOSE_CONV: tfp.Convolution.load,
            # BuiltinOperator.SQRT : None
            BuiltinOperator.MEAN: tfp.Mean.load,
            BuiltinOperator.SPACE_TO_DEPTH: tfp.SpaceToDepth.load,
        }

        parsedLayers = []
        version = self.model.Version()
        description = self.model.Description()
        buffers = [
            self.model.Buffers(idx) for idx in range(
                self.model.BuffersLength())]
        subgraphs = [
            self.model.Subgraphs(idx) for idx in range(
                self.model.SubgraphsLength())]
        operator_codes = [
            self.model.OperatorCodes(idx) for idx in range(
                self.model.OperatorCodesLength())]

        if len(subgraphs) != 1:
            raise ValueError(
                "Only a single subgraph is supported in the current TFLite")

        sg = subgraphs[0]
        tensors = [sg.Tensors(idx) for idx in range(sg.TensorsLength())]
        input_idxs = [sg.Inputs(idx) for idx in range(sg.InputsLength())]
        output_idxs = [sg.Outputs(idx) for idx in range(sg.OutputsLength())]
        operators = [sg.Operators(idx) for idx in range(sg.OperatorsLength())]

        self.tensor_index_map = {str(tensors[idx].Name().decode(
            "utf-8")): idx for idx in range(len(tensors))}

        if len(output_idxs) != 1:
            raise ValueError("Unsupported more than one output")
        if len(input_idxs) != 1:
            raise ValueError("Unsupported more than one input")

        for obj in operators:
            op_type = operator_codes[obj.OpcodeIndex()].BuiltinCode()
            opParser = subParsers.get(op_type, None)

            compiler_assert(opParser is not None,
                            ErrorTable.StageTypeNotSupported, op_type)
            if opParser is not None:
                parsedLayers.extend(opParser(obj, op_type, tensors, buffers))

        # Mangle tensor names
        tensorNamesDict = OrderedDict()
        for layer in parsedLayers:
            for tensorName in list(layer.getInputTensorNames()) + \
                    list(layer.getOutputTensorNames()):
                if tensorName not in tensorNamesDict:
                    tensorNamesDict[tensorName] = MangledName(
                        OriginalName(tensorName))

        # Replace tensor names in layers with mangled ones:
        for layer in parsedLayers:
            layer.setInputTensorNames([tensorNamesDict[name]
                                       for name in layer.getInputTensorNames()])
            layer.setOutputTensorNames(
                [tensorNamesDict[name] for name in layer.getOutputTensorNames()])

        # Convert inPlace operations into regular ones
        parsedLayers = regularizeInPlaceOps(parsedLayers)

        # Create tensor objects for each operation
        createTensors(parsedLayers)

        # Add quantization fields to quantized tensors
        quantizeTensors(parsedLayers, tensors)

        # Introduce Output operations
        output_tensor_name = str(tensors[output_idxs[0]].Name().decode("utf-8"))
        input_tensor_name = str(tensors[input_idxs[0]].Name().decode("utf-8"))
        parsedLayers = insertInputOutputOps(
            parsedLayers, input_tensor_name, output_tensor_name)

        print("Fusing Add and Batch after Convolution")
        g = buildGraph(parsedLayers)
        parsedLayers = buildLayerLists(g)

        return parsedLayers

    def get_layer_data(self, outBlobName):

        outBlobName = str(outBlobName)

        if outBlobName not in self.tensor_index_map.keys():
            raise ValueError("Impossible to get tensor {}".format(outBlobName))

        out_data = self.interpreter.get_tensor(
            self.tensor_index_map[outBlobName])

        if (len(out_data.shape) == 4):
            out_data = out_data.transpose((0, 3, 1, 2))
        elif (len(out_data.shape) == 2):
            out_data = out_data.reshape((1,1,out_data.shape[0],out_data.shape[1]))
        return out_data

