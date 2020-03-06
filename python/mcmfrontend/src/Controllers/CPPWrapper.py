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

from Models.EnumDeclarations import Parser, ErrorTable
import re
import json
import numpy as np
import networkx as nx
from Controllers.EnumController import throw_error
from Controllers.GraphUtils import buildGraph
import os
import sys
import Models.Layouts as Layouts
try:
    from Controllers import mcmpathinclude  # noqa: E261
    assert mcmpathinclude  # silence pyflakes
    import composition_api as ca
except ImportError:
    print("problem importing mcmpathinclude")

convolution_node_id = 0
fully_node_id = 0
pooling_node_id = 0
relu_node_id = 0
leaky_relu_node_id = 0
dropout_node_id = 0
softmax_node_id = 0
eltwise_node_id = 0
concat_node_id = 0
depthwise_node_id = 0
scale_node_id = 0
sigmoid_node_id = 0
minimum_node_id = 0
maximum_node_id = 0
power_node_id = 0
reorgYolo_node_id = 0
text = re.compile(r"\w*[Tt]\w*[Ee]\w*[Xx]\w*[Tt]")
binary = re.compile(r"\w*[Bb]\w*[Ii]\w*[Nn]\w*[Aa]\w*[Rr]\w*[Yy]")
none = re.compile(r"\w*[Nn]\w*[Oo]\w*[Nn]\w*[Ee]")


def initialize_execution_file(weights="None"):

    exec_file = open("templateExample.cpp", "w+")
    exec_file.write(
        '//This file is the parsed network which is created through python.\n')
    exec_file.write(
        '#include ' +
        '"' +
        'include/mcm/compiler/compilation_unit.hpp' +
        '"\n')
    exec_file.write(
        '#include ' +
        '"' +
        'include/mcm/utils/data_generator.hpp' +
        '"\n')
    exec_file.write('#include ' + '"' +
                    'include/mcm/op_model.hpp' + '"\n')
    exec_file.write(
        '#include ' +
        '"' +
        'include/mcm/utils/hardware_tests.hpp' +
        '"\n\n')

    exec_file.write('#include ' + '<' + 'iostream' + '>\n')
    exec_file.write('#include ' + '<' + 'fstream' + '>\n\n')

    if (text.match(weights)):

        exec_file.write(
            'template <typename T> std::vector<T> read_weights_from_file(std::string input_file)\n{\n')
        exec_file.write(' ' * 4 + 'std::ifstream file;\n')
        exec_file.write(' ' * 4 + 'T inputString;\n')

        exec_file.write(' ' * 4 + 'std::vector<T> data;\n')
        exec_file.write(' ' * 4 + 'while(file>>inputString)\n')
        exec_file.write(' ' * 8 + 'data.push_back(inputString);\n')

        exec_file.write(' ' * 4 + 'file.close();\n')
        exec_file.write(' ' * 4 + 'return data;\n}\n\n')

    elif (binary.match(weights)):
        exec_file.write(
            'template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)\n{\n')
        exec_file.write(
            ' ' *
            4 +
            'std::ifstream file(input_file, std::ifstream::binary);\n')
        exec_file.write(' ' * 4 + 'T2 inputString;\n')

        exec_file.write(' ' * 4 + 'std::vector<T2> data;\n')
        exec_file.write(
            ' ' *
            4 +
            'while(file.read(reinterpret_cast<char*>(&inputString), sizeof(T2)))\n')
        exec_file.write(' ' * 8 + 'data.push_back(inputString);\n')

        exec_file.write(' ' * 4 + 'file.close();\n')
        exec_file.write(
            ' ' *
            4 +
            'std::vector<T1> return_data(data.begin(), data.end());\n')
        exec_file.write(' ' * 4 + 'return return_data;\n}\n\n')

    exec_file.write('int main()\n{\n')
    exec_file.write(
        ' ' *
        4 +
        'double inf = std::numeric_limits<double>::infinity();\n\n')

    exec_file.write(' ' * 4 + 'mv::CompilationUnit unit("parserModel");\n')
    exec_file.write(' ' * 4 + 'mv::OpModel& om = unit.model();\n')
    return exec_file


def finalize_execution_file(exec_file, comp_descriptor):

    exec_file.write(
        ' ' *
        4 +
        'std::string compDescPath = "' + str(comp_descriptor) + '";\n')
    exec_file.write(
        ' ' *
        4 +
        'unit.loadCompilationDescriptor(compDescPath);\n\n')

    exec_file.write(
        ' ' *
        4 +
        'unit.loadTargetDescriptor(mv::Target::ma2490);\n')
    exec_file.write(' ' * 4 + 'unit.initialize();\n')
    exec_file.write(' ' * 4 + 'unit.run();\n}\n')

    exec_file.close()
    return exec_file

def composeForCpp(parsedLayers, arguments):

    reference_list = {}
    target_desc = "ma2490"
    comp_unit = ca.getCompilationUnit(target_desc)
    mcm_file = None
    produce_network_weights_mcm = None
    om = ca.getModel(comp_unit)
    g = buildGraph(parsedLayers)

    if arguments.comp_descriptor is not None:
        comp_desc_file = arguments.comp_descriptor
    else:
        sys.exit('Please provide a compilation Descriptor!')

    ca.loadCompilationDescriptor(comp_unit, comp_desc_file)
    json_file = json.load(open(comp_desc_file))
    try:
        produce_network_description_mcm = json_file['initialize']['Singular'][0]['recorded_model']
    except KeyError:
        produce_network_description_mcm = False
    try:
        produce_network_weights_mcm = json_file['initialize']['Singular'][0]['weights_form']
    except KeyError:
        produce_network_weights_mcm = "None"

    if (produce_network_description_mcm):
        if (text.match(produce_network_weights_mcm)
                or binary.match(produce_network_weights_mcm)):
            mcm_file = initialize_execution_file(produce_network_weights_mcm)
        else:
            mcm_file = initialize_execution_file()

    tensor_mapping_dict = {}
    for index, child in enumerate(nx.lexicographical_topological_sort(g)):
        layer = g.node[child]['ref']
        n = layer.name.stringifyName()
        reference_list[n], om = buildOM(g, child, reference_list, om,
                                        arguments.parser, mcm_file, tensor_mapping_dict, produce_network_weights_mcm)
        if (produce_network_description_mcm):
            print('Layer ' + str(layer) + ' is parsed')

    if (produce_network_description_mcm):
        mcm_file = finalize_execution_file(mcm_file, comp_desc_file)

    ca.compile(comp_unit)
    ca.deleteCompilationUnitObject(comp_unit)

def writeInputBin(parsedLayers, emulator):
    from Controllers.Parsers.Parser.Input import Input
    from Controllers.Parsers.Parser.Convolution2D import Convolution2D
    g = buildGraph(parsedLayers)
    for index, child in enumerate(nx.lexicographical_topological_sort(g)):
        layer = g.node[child]['ref']
        if isinstance(layer, Input):
            output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
            tensor_data = emulator.get_layer_data_cpp(output_tensor_name)
            next_op_from_input = find_next_op(g, layer, layer.getOutputTensors()[0])
            layout = Layouts.ZMajor
            # if (isinstance(next_op_from_input, Convolution2D) and (next_op_from_input.getInputTensors()[0].getShape()[1] < 16)):
            #     layout = Layouts.ChannelMajor
            tensor_data = np.transpose(tensor_data, layout)
            write_tensor(layer.getOutputTensors()[0], tensor_data, layout)

def write_tensor(tensor, data, layout):
    directory = 'output/'
    name = tensor.name.stringifyOriginalName().replace("/", "_")
    fp = open("output/input.dat".format(directory, name), 'wb')
    fp.write((data.flatten()).astype(tensor.dtype).data)
    fp.close()
    print(
        "Generate tensor {}.bin (layout {}, dtype {})".format(
            name, layout, tensor.dtype))

def get_parse_quant(tensor):
    try:
        tensor_quant = tensor.getQuantizationParameters()
    except AttributeError:
        empty_quant_params = ca.getQuantParams(
            ca.getData(np.array([], dtype=np.int64)),
            ca.getData(np.array([], dtype=np.float64)),
            ca.getData(np.array([], dtype=np.float64)),
            ca.getData(np.array([], dtype=np.float64)))
        return (empty_quant_params, ca.getData(np.array([], dtype=np.int64)),
                ca.getData(np.array([], dtype=np.float64)),
                ca.getData(np.array([], dtype=np.float64)),
                ca.getData(np.array([], dtype=np.float64)))
    try:
        zero_quant_data = tensor_quant.getZeroPoint().astype(np.int64)
        zero_data = ca.getData(zero_quant_data)
    except ValueError:
        zero_quant_data = np.array([], dtype=np.int64)
        zero_data = ca.getData(np.array([], dtype=np.int64))

    try:
        scale_quant_data = tensor_quant.getScale().astype(np.float64).flatten()
        #TENSORFLOW CRETES A 2-D Quant array, from Compiler point
        #always 1-dim, worst case scenario 1-value per OC
        scale_data = ca.getData(scale_quant_data)
    except ValueError:
        scale_quant_data = np.array([], dtype=np.float64)
        scale_data = ca.getData(np.array([], dtype=np.float64))

    try:
        min_quant_param = (np.min(tensor.quantization.float_range))
        min_quant_param = np.asarray([min_quant_param])
        min_quant_data = ca.getData(min_quant_param)
    except ValueError:
        min_quant_param = np.array([-np.inf], dtype=np.float64)
        min_quant_data = ca.getData(np.array([-np.inf], dtype=np.float64))

    try:
        max_quant_param = np.max(tensor.quantization.float_range)
        max_quant_param = np.asarray([max_quant_param])
        max_quant_data = ca.getData(max_quant_param)
    except ValueError:
        max_quant_param = np.array([np.inf], dtype=np.float64)
        max_quant_data = ca.getData(np.array([np.inf], dtype=np.float64))

    mv_quant_params = ca.getQuantParams(
        zero_data, scale_data, min_quant_data, max_quant_data)

    return mv_quant_params, zero_quant_data, scale_quant_data, min_quant_param, max_quant_param


def call_recored_weights(recorded_type, array, weight_tensor_name):

    path = 'weights_bias'
    # first layer with bias and weigts
    if (binary.match(recorded_type) or text.match(recorded_type)) and (
            convolution_node_id + depthwise_node_id + scale_node_id + fully_node_id == 0):
        try:
            os.mkdir("weights_bias")
        except FileExistsError:
            pass
    if (text.match(recorded_type)):
        filename = path + "/" + str(weight_tensor_name) + ".txt"
        np.savetxt(filename, array.flatten(), '%.3d')
    elif (binary.match(recorded_type)):
        filename = path + "/" + str(weight_tensor_name) + ".dat"
        array.flatten().tofile(filename, "")
    else:
        sys.exit("Provide weights in text or binary form.")


def dir_c_type(type_value):

    if isinstance(type_value(), int):
        return 'int64_t'
    else:
        return 'double'


def weight_buffer_type(type_value):
    if isinstance(type_value(), int):
        return 'uint8_t'
    else:
        return 'half'


def bias_buffer_type(type_value):
    if isinstance(type_value(), int):
        return 'int32_t'
    else:
        return 'half'

def find_next_op(g, layer, output_tensor):
    from Controllers.Parsers.Parser.Input import Input
    found = layer
    for index, child in enumerate(nx.lexicographical_topological_sort(g)):
        layer = g.node[child]['ref']
        if (isinstance(layer, Input)):
            continue
        if (layer.getInputTensors()[0] == output_tensor):
            found = layer
    return found

def buildOM(
        g,
        gnode_name,
        reflist,
        om,
        parser,
        output_file=None,
        tensor_mapping_dict={},
        keep_weights="None"):
    # """
    #     Construct C++ Representation of a Layer.
    #     Return the iterator for passing forward
    # """

    gnode = g.node[gnode_name]
    layer = gnode['ref']

    from Controllers.Parsers.Parser.InnerProduct import InnerProduct
    from Controllers.Parsers.Parser.Input import Input
    from Controllers.Parsers.Parser.Output import Output
    from Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
    from Controllers.Parsers.Parser.Pooling import Pooling
    from Controllers.Parsers.Parser.Eltwise import Eltwise
    from Controllers.Parsers.Parser.ReLU import ReLU, LeakyReLU
    from Controllers.Parsers.Parser.Sigmoid import Sigmoid
    from Controllers.Parsers.Parser.Power import Power
    from Controllers.Parsers.Parser.Minimum import Minimum
    from Controllers.Parsers.Parser.Maximum import Maximum
    # from Controllers.Parsers.Parser.PReLU import PReLU
    from Controllers.Parsers.Parser.Concat import Concat
    from Controllers.Parsers.Parser.Scale import Scale
    from Controllers.Parsers.Parser.Bias import Bias
    from Controllers.Parsers.Parser.BatchNorm import BatchNorm
    from Controllers.Parsers.Parser.Softmax import Softmax
    from Controllers.Parsers.Parser.SpaceToDepth import SpaceToDepth
    from Controllers.Parsers.Parser.NoOp import NoOp

    _ref = None

    type_dict = {
        np.uint8: np.int,
        np.dtype('uint8'): np.int,
        np.dtype('int32'): np.int,
        np.float64: np.float64,
        np.float16: np.float64,
        np.dtype('float16'): np.float64
    }

    sw_layers_type_dict = {
        np.uint8: np.uint8,
        np.float16: np.float16,
    }

    order_type_dict = {
        np.int: "UInt8",
        np.float16: "Float64",
        np.float64: "Float64"
    }

    sw_layers_order_type_dict = {
        np.uint8: "UInt8",
        np.float16: "Float16"
    }

    clip_registers = {
        np.float32: np.dtype('double'),
        np.int32: np.dtype('int64'),
        np.int: np.dtype('int64')
    }

    mcm_4d_layout = {
        Parser.Caffe: "NCHW",
        Parser.TensorFlowLite: "NHWC",
        Parser.TensorFlow: "NHWC"
    }

    mcm_1d_layout = {
        Parser.Caffe: "W",
        Parser.TensorFlowLite: "W",
        Parser.TensorFlow: "W"
    }

    mcm_2d_layout = {
        Parser.Caffe: "HW",
        Parser.TensorFlowLite: "WC",
        Parser.TensorFlow: "WC"
    }

    constant_type = {
        'int64_t': 'constantInt',
        'double': 'constant'
    }

    minimum_type = {
        np.dtype('int64') : 'minimumInt',
        np.float64 : 'minimumDouble'
    }

    maximum_type = {
        np.dtype('int64') : 'maximumInt',
        np.float64 : 'maximumDouble'
    }

    global convolution_node_id, pooling_node_id, relu_node_id, leaky_relu_node_id, \
        dropout_node_id, eltwise_node_id, softmax_node_id, depthwise_node_id,   \
        concat_node_id, scale_node_id, fully_node_id, sigmoid_node_id, power_node_id, \
        minimum_node_id, maximum_node_id, reorgYolo_node_id

    if isinstance(layer, Bias):
        assert 0, "Standalone Bias Layer currently unsupported by C++ API"

    if isinstance(layer, Input):
        s = layer.getOutputTensors()[0].getShape()  # getShape returns N, C, H, W
        shape = ca.getShape(s[3], s[2], s[1], s[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        _ref = ca.input(om, shape, type_value, ca.getOrder(mcm_4d_layout[parser]), get_parse_quant(layer.getOutputTensors()[0])[0], output_tensor_name)
        
        if (output_file is not None):
            output_file.write(
                ' ' *
                4 +
                'auto input0 = om.input({' +
                str(s[3]) +
                ',' +
                str(s[2]) +
                ',' +
                str(s[1]) +
                ',' + str(s[0]) +
                '}, mv::DType("' +
                order_type_dict[type(type_value)] +
                '"), ' +
                'mv::Order::getZMajorID(4), ' +
                '{{' +
                ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                '},{' +
                ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                '},{' +
                ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                '},{' +
                ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                '}}, "' +
                str(output_tensor_name) +
                '");\n\n')
            tensor_mapping_dict[output_tensor_name] = 'input0'

    elif isinstance(layer, Output):
        pred = list(g.predecessors(gnode_name))
        _ref = ca.output(om, reflist[pred[0]])

        if (output_file is not None):
            input_tensor = tensor_mapping_dict[layer.getInputTensors()[
                0].getName().stringifyName()]
            output_file.write(
                ' ' * 4 + 'om.output(' + str(input_tensor) + ');\n\n')

    elif isinstance(layer, Pooling):
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        ry, rx = layer.getKernelSize()
        sy, sx = layer.getStride()
        py, px = layer.getPadding()
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])

        if layer.getType() == Pooling.Type.MAX:
            if (parser == Parser.Caffe):
                _ref = ca.maxpool2D_caffe(
                    om, in_, rx, ry, sx, sy, px[0], py[0], output_tensor_name)
            elif (parser == Parser.TensorFlowLite or parser == Parser.TensorFlow):
                _ref = ca.maxpool2D(om, in_, rx, ry, sx, sy,
                                    px[0], px[1], py[0], py[1], str(order_type_dict[type(type_value)]), mv_quant_params[0], output_tensor_name)
            else:
                throw_error(ErrorTable.ParserNotSupported, parser.name)
        elif layer.getType() == Pooling.Type.AVE:
            if (parser == Parser.Caffe):
                _ref = ca.avgpool2D_caffe(
                    om, in_, rx, ry, sx, sy, px[0], py[0], output_tensor_name)
            elif (parser == Parser.TensorFlowLite or parser == Parser.TensorFlow):
                _ref = ca.avgpool2D(
                    om, in_, rx, ry, sx, sy, px[0], px[1], py[0], py[1], order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)
            else:
                throw_error(ErrorTable.ParserNotSupported, parser.name)

        if (output_file is not None):

            pool_mcm = {
                Pooling.Type.MAX: 'maxPool',
                Pooling.Type.AVE: 'averagePool'
            }

            output_file.write(' ' * 4 + 'auto pool' + str(pooling_node_id) + ' = om.' + str(pool_mcm[layer.getType()]) + '(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) + ', {' + str(ry) +
                              ', ' + str(rx) + '}, {' + str(sy) + ', ' + str(sx) + '}, {' + str(px[0]) + ', ' + str(px[1]) +
                              ', ' + str(py[0]) + ', ' + str(py[1]) + '}, ' + 'true, ' + 'mv::DType("' +
                              str(order_type_dict[type(type_value)]) + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'pool' + \
                str(pooling_node_id)
            pooling_node_id += 1

    elif isinstance(layer,LeakyReLU):
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        ##NOTE: THIS NEEDS TO BE USED FOR LRELU ON PPE TASKS
        ##(layer.reluX)
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        alpha = layer.getAlpha()

        _ref = ca.leaky_relu(om, in_, alpha, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto leakyRelu' + str(leaky_relu_node_id) + ' = om.leakyRelu(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) + ', ' + str(alpha) +
                              ', mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'leakyRelu' + \
                str(leaky_relu_node_id)
            leaky_relu_node_id += 1

    elif isinstance(layer,Sigmoid):
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        ##NOTE: THIS NEEDS TO BE USED FOR Sigmoid ON PPE TASKS
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        _ref = ca.sigmoid(om, in_, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto sigmoid' + str(sigmoid_node_id) + ' = om.sigmoid(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'sigmoid' + \
                str(sigmoid_node_id)
            sigmoid_node_id += 1

    elif isinstance(layer, Minimum):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        minimum = layer.getMinimum()
        minimum = minimum.astype(clip_registers[type(minimum)])
        ##NOTE: THIS NEEDS TO BE USED FOR Sigmoid ON PPE TASKS
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        _ref = ca.minimum(om, in_, minimum, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)
        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto minimum' + str(minimum_node_id) + ' = om.' + str(minimum_type[type(minimum)]) + '(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', '+ str(minimum) +', mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'minimum' + \
                str(minimum_node_id)
            minimum_node_id += 1

    elif isinstance(layer, Maximum):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        maximum = layer.getMaximum()
        maximum = maximum.astype(clip_registers[type(maximum)])
        ##NOTE: THIS NEEDS TO BE USED FOR Sigmoid ON PPE TASKS
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        _ref = ca.maximum(om, in_, maximum, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto maximum' + str(maximum_node_id) + ' = om.' + str(maximum_type[type(maximum)]) + '(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ',' + str(maximum) + ', mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'maximum' + \
                str(maximum_node_id)
            maximum_node_id += 1

    elif isinstance(layer, ReLU):
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        ##NOTE: THIS NEEDS TO BE USED FOR RELUX ON PPE TASKS
        ##(layer.reluX)
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        _ref = ca.relu(om, in_, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto relu' + str(relu_node_id) + ' = om.relu(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'relu' + \
                str(relu_node_id)
            relu_node_id += 1

    elif isinstance(layer, Power):
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        ##NOTE: THIS NEEDS TO BE USED FOR RELUX ON PPE TASKS
        ##(layer.reluX)
        pred = list(g.predecessors(gnode_name))
        in1_ = reflist[pred[0]]

        if len(pred) == 1:
            in2_ = reflist[pred[0]]  # Same input.
        else:
            in2_ = reflist[pred[1]]


        _ref = ca.power(om, in1_, in2_, order_type_dict[type(type_value)], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto power' + str(power_node_id) + ' = om.power({' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', ' + str(tensor_mapping_dict[layer.getInputTensors()[1].getName().stringifyName()]) + '}, '
                              + 'mv::DType("' + order_type_dict[type(type_value)] + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'power' + \
                str(power_node_id)
            power_node_id += 1

    elif isinstance(layer, NoOp):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        _ref = ca.identity(
            om, in_, "Float16", mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto identity' + str(dropout_node_id) + ' = om.identity(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', mv::DType("' + "Float16" + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'identity' + \
                str(dropout_node_id)
            dropout_node_id += 1

    elif isinstance(layer, Eltwise):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)

        pred = list(g.predecessors(gnode_name))
        in1_ = reflist[pred[0]]

        if layer.getType() == Eltwise.Type.WSUM:

            if len(pred) == 1:
                in2_ = reflist[pred[0]]  # Same input.
            else:
                in2_ = reflist[pred[1]]


        elif layer.getType() == Eltwise.Type.WPROD:

            in2_ = reflist[pred[1]]

        eltwise_map = {
            Eltwise.Type.WSUM: 'Add',
            Eltwise.Type.WPROD: 'Multiply',
        }


        _ref = ca.eltwise(om, in1_, in2_, eltwise_map[layer.getType()], order_type_dict[type(type_value)],
                      mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            output_file.write(' ' * 4 + 'auto eltwise' + str(eltwise_node_id) + ' = om.eltwise({' +
                               str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ',' + str(tensor_mapping_dict[layer.getInputTensors()[1].getName().stringifyName()]) + '}, "' +
                               eltwise_map[layer.getType()] +
                              '", mv::DType("' + str(order_type_dict[type(type_value)]) + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'eltwise' + \
                str(eltwise_node_id)
            eltwise_node_id += 1

    #RECORDED MODEL NEEDS TO BE FIXED AND THE MIXED PRECISION
    elif isinstance(layer, Scale) or isinstance(layer, BatchNorm):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        scale_data = layer.getMultiplier()
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)
        scale_vector = ca.getData(scale_data.astype(
            type_dict[layer.getMultiplier().dtype]))
        scale_type_value = type(type_dict[layer.getMultiplier().dtype](0.0))
        scale_tensor_name = layer.getMultiplier().getName().stringifyName()
        scale_mv_quant_params = get_parse_quant(layer.getMultiplier())

        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()

        scale_param = ca.constant(om, scale_vector, ca.getShape(scale_data.shape[0]), ca.getOrder(
            mcm_4d_layout[parser]), type_value, scale_mv_quant_params[0], scale_tensor_name)
        scale = ca.scale(
            om, in_, scale_param, mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            if (keep_weights != "None"):
                call_recored_weights(
                    keep_weights,
                    scale_data,
                    layer.getMultiplier().getName().stringifyName())
                output_file.write(' ' *
                                  4 +
                                  'std::vector<' +
                                  dir_c_type(scale_type_value) +
                                  '> weightsData' +
                                  str(scale_node_id) +
                                  ' = read_weights_from_file<' +
                                  dir_c_type(scale_type_value) +
                                  '>(path + "/projects/Fathom/src2/weights_bias/' +
                                  str(layer.getMultiplier().getName().stringifyName()) +
                                  '.dat");\n')
            else:
                output_file.write(' ' *
                                  4 +
                                  'std::vector<' +
                                  dir_c_type(scale_type_value) +
                                  '> weightsData' +
                                  str(scale_node_id) +
                                  ' = mv::utils::generateSequence<' +
                                  dir_c_type(scale_type_value) +
                                  '> ({}*{}*{}*{});\n'.format(scale_data.shape[0]))

            output_file.write(' ' *
                              4 +
                              'auto weights' +
                              str(scale_node_id) +
                              ' = om.' +
                              constant_type[dir_c_type(scale_type_value)] +
                              '(weightsData' +
                              str(scale_node_id) +
                              ',{' +
                              str(scale_data.shape[0]) +
                              '}, mv::DType("' +
                              order_type_dict[scale_type_value] +
                              '"), ' +
                              'mv::Order::getColMajorID(1), ' +
                              '{{' +
                              ', '.join(map(str, get_parse_quant(layer.getMultiplier())[1])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getMultiplier())[2])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getMultiplier())[3])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getMultiplier())[4])) +
                              '}}, "' +
                              str(scale_tensor_name) +
                              '");\n')

            output_file.write(' ' *
                              4 +
                              'auto scale' +
                              str(scale_node_id) +
                              ' = om.scale(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', weights' +
                              str(scale_node_id) +
                              ', {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                              '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                              '}}, "' +
                              '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'scale' + \
                str(scale_node_id)

        if layer.hasBiasBeta():
            bias_data = layer.getBiasBeta()
            bias_type_value = type(type_dict[layer.getBias().dtype](0.0))
            bias_vector = ca.getData(
                bias_data.astype(type_dict[layer.getBias().dtype]))
            bias_tensor_name = layer.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(layer.getBias())

            bias_param = ca.constant(om, bias_vector, ca.getShape(bias_data.shape[0]), ca.getOrder(
                mcm_1d_layout[parser]), bias_mv_quant_params[0], bias_tensor_name)
            bias = ca.bias(
                om, scale, bias_param, bias_mv_quant_params[0], bias_tensor_name)

            if (output_file is not None):
                if (keep_weights != "None"):
                    call_recored_weights(
                        keep_weights, bias_data, layer.getBias().getName().stringifyName())
                    output_file.write(' ' *
                                      4 +
                                      'std::vector<' +
                                      dir_c_type(bias_type_value) +
                                      '> biasWeightsData' +
                                      str(scale_node_id) +
                                      ' = read_weights_from_file<' +
                                      dir_c_type(bias_type_value) +
                                      '>(path + "/projects/Fathom/src2/weights_bias/' +
                                      str(layer.getBias().getName().stringifyName()) +
                                      '.dat");\n')
                else:
                    output_file.write(
                        ' ' *
                        4 +
                        'std::vector<' +
                        dir_c_type(bias_type_value) +
                        '> biasWeightsData' +
                        str(scale_node_id) +
                        ' = mv::utils::generateSequence<' +
                        dir_c_type(bias_type_value) +
                        '> ({});\n'.format(
                            bias_data.shape[0]))

                output_file.write(' ' *
                                  4 +
                                  'auto biasWeights' +
                                  str(scale_node_id) +
                                  ' = om.' +
                                  constant_type[dir_c_type(bias_type_value)] +
                                  '(biasWeightsData' +
                                  str(scale_node_id) +
                                  ',{' +
                                  str(bias_data.shape[0]) +
                                  '}, mv::DType("' +
                                  order_type_dict[bias_type_value] +
                                  '"), ' +
                                  'mv::Order::getColMajorID(1), ' +
                                  '{{' +
                                  ', '.join(map(str, get_parse_quant(layer.getBiasBeta())[1])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getBiasBeta())[2])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getBiasBeta())[3])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getBiasBeta())[4])) +
                                  '}}, "' +
                                  str(bias_tensor_name) +
                                  '");\n')

                output_file.write(' ' *
                                  4 +
                                  'auto bias_s' +
                                  str(scale_node_id) +
                                  ' = om.bias(' +
                                  str(tensor_mapping_dict[output_tensor_name]) +
                                  ', biasWeights' +
                                  str(scale_node_id) +
                                  ', {{' +
                                  ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                                  '},{' +
                                  ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                                  '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_i' + \
                    str(scale_node_id)

            _ref = bias
        else:
            _ref = scale
        scale_node_id += 1

    # elif isinstance(layer, BatchNorm):

    #     pred = list(g.predecessors(gnode_name))
    #     in_ = reflist[pred[0]]

    #     '''mean_data = ca.getData(layer.getMean().astype(np.float64))
    #     mean_param = ca.constant(om, mean_data, ca.getShape(*layer.getMean().shape))

    #     var_data = ca.getData(layer.getVariance().astype(np.float64))
    #     variance_param = ca.constant(om, var_data, ca.getShape(*layer.getVariance().shape))

    #     offset_data = ca.getData(layer.getBiasBeta().astype(np.float64))
    #     offset_param = ca.constant(om, offset_data, ca.getShape(*layer.getBiasBeta().shape))

    #     scale_data = ca.getData(layer.getMultiplier().astype(np.float64))
    #     scale_param = ca.constant(om, scale_data, ca.getShape(*layer.getMultiplier().shape))

    #     eps = layer.getEPS()

    # _ref = ca.batchNorm(om, in_, mean_param, variance_param, offset_param,
    # scale_param, eps)'''

    #     scale_data = layer.getMultiplier()
    #     scale_vector = ca.getData(scale_data.astype(np.float64))
    #     scale_param = ca.constant(om, scale_vector, ca.getShape(scale_data.shape[0]))
    #     scale = ca.scale(om, in_, scale_param)

    #     bias_data = layer.getBiasBeta()
    #     bias_vector = ca.getData(bias_data.astype(np.float64))
    #     bias_param = ca.constant(om, bias_vector, ca.getShape(bias_data.shape[0]))
    #     bias = ca.bias(om, scale, bias_param)
    #     _ref = bias

    elif isinstance(layer, Softmax):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        type_value = type_dict[layer.getOutputTensors()[0].dtype](0.0)

        _ref = ca.softmax(
            om, in_, mv_quant_params[0], output_tensor_name)

        if (output_file is not None):
            output_file.write(' ' * 4 + 'auto softmax' + str(softmax_node_id) + ' = om.softmax(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', "C", ' + ',  mv::DType(' + str(order_type_dict[type(type_value)]) + '), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                              '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'softmax' + \
                str(softmax_node_id)
            softmax_node_id += 1

    elif isinstance(layer, SpaceToDepth):

        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        tensor_dtype = layer.getOutputTensors()[0].dtype
        type_value = sw_layers_type_dict[tensor_dtype]

        stride = layer.getBlockSize()

        _ref = ca.reorgYolo(
            om, in_, stride, "Float16", mv_quant_params[0], output_tensor_name)

        if (output_file is not None):
            output_file.write(' ' * 4 + 'auto reorgYolo' + str(reorgYolo_node_id) + ' = om.reorgYolo(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', ' + str(stride) +
                              ', mv::DType("' + "Float16" + '"), {{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                              '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'reorgYolo' + \
                str(reorgYolo_node_id)
            reorgYolo_node_id += 1

    elif isinstance(layer, InnerProduct):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        w_data = layer.getWeights().data
        w_orig = w_data

        w_data = np.transpose(w_data, (3, 2, 1, 0))
        arr = w_data.flatten()

        weight_tensor_name = layer.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(layer.getWeights())
        weight_type_value = type(type_dict[layer.getWeights().dtype](0.0))
        output_type_value = type(type_dict[layer.getOutputTensors()[0].dtype](0.0))
        weights_data = ca.getData(np.array(arr).astype(weight_type_value))
        # size of output channels is w[2]
        # other dimension of weights is the mult of all the input tensor
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        weights_ = ca.constant(om, weights_data, ca.getShape(w_orig.shape[1], (w_orig.shape[2] * w_orig.shape[3] * w_orig.shape[0])),
            ca.getOrder(mcm_2d_layout[parser]), weight_mv_quant_params[0], weight_tensor_name)
        fc = ca.fullyConnected(om, in_, weights_, order_type_dict[output_type_value], mv_quant_params[0], output_tensor_name)
        if (output_file is not None):
            if (keep_weights != "None"):
                call_recored_weights(keep_weights, w_orig,
                                     layer.getWeights().getName().stringifyName())
                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) + '> weightsData' +
                                  str(fully_node_id) + ' = read_weights_from_file<' +
                                  dir_c_type(weight_type_value) +
                                  '>(path + "/projects/Fathom/src2/weights_bias/' +
                                  str(layer.getWeights().getName().stringifyName()) + '.dat");\n')
            else:
                output_file.write(' ' * 4 + 'std::vector<' +
                                  dir_c_type(weight_type_value) +
                                  '> weightsData' + str(fully_node_id) +
                                  ' = mv::utils::generateSequence<' +
                                  dir_c_type(weight_type_value) +
                                  '> ({}*{});\n'.format(w_orig.shape[1], w_orig.shape[2] * w_orig.shape[3] * w_orig.shape[0]))

            output_file.write(' ' * 4 + 'auto weights' + str(fully_node_id) + ' = om.' + constant_type[dir_c_type(weight_type_value)] +
                              '(weightsData' + str(fully_node_id) + ',{' + str(w_orig.shape[1]) +
                              ',' + str(w_orig.shape[2] * w_orig.shape[3] * w_orig.shape[0]) +
                              '}, mv::DType("' + order_type_dict[weight_type_value] + '"), ' + 'mv::Order("' + mcm_2d_layout[parser] + '"), ' +
                              '{{' + ', '.join(map(str, get_parse_quant(layer.getWeights())[1])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getWeights())[2])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getWeights())[3])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getWeights())[4])) +
                              '}}, "' +  str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto fc' + str(fully_node_id) + ' = om.fullyConnected(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', weights' + str(fully_node_id) + ', mv::DType("' + order_type_dict[output_type_value] + '")' +
                              ', {{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'fc' + str(fully_node_id)

        if layer.biasEnabled():
            b = layer.getBias()
            bias_tensor_name = layer.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(layer.getBias())
            bias_type_value = type(type_dict[layer.getBias().dtype](0.0))
            output_type_value = type(type_dict[layer.getOutputTensors()[0].dtype](0.0))

            bias_data = ca.getData(np.array(b.data.flatten()).astype(bias_type_value))
            bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]), ca.getOrder(mcm_1d_layout[parser]), bias_mv_quant_params[0],
                bias_tensor_name + "weights")
            _ref = ca.bias(om, fc, bias, order_type_dict[output_type_value], mv_quant_params[0], bias_tensor_name)

            if (output_file is not None):

                if (keep_weights != "None"):
                    call_recored_weights(keep_weights, b.data, layer.getBias().getName().stringifyName())
                    output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) +
                                      '> biasWeightsData' + str(fully_node_id) +
                                      ' = read_weights_from_file<' + dir_c_type(bias_type_value) +
                                      '>(path + "/projects/Fathom/src2/weights_bias/' +
                                      str(layer.getBias().getName().stringifyName()) + '.dat");\n')

                else:
                    output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) +
                                      '> biasWeightsData' + str(fully_node_id) +
                                      ' = mv::utils::generateSequence<' + dir_c_type(bias_type_value) + '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(fully_node_id) +
                                  ' = om.' + constant_type[dir_c_type(bias_type_value)] +
                                  '(biasWeightsData' + str(fully_node_id) +
                                  ',{' + str(b.shape[0]) + '}, mv::DType("' +
                                  order_type_dict[bias_type_value] +
                                  '"), ' + 'mv::Order::getColMajorID(1), ' +
                                  '{{' + ', '.join(map(str, get_parse_quant(layer.getBias())[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[4])) +
                                  '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_i' + str(fully_node_id) + ' = om.bias(' +
                                  str(tensor_mapping_dict[output_tensor_name]) + ', biasWeights' + str(fully_node_id) +
                                  ', mv::DType("' + order_type_dict[output_type_value] + '")'
                                  ', {{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}});\n\n')


                tensor_mapping_dict[output_tensor_name] = 'bias_i' + \
                    str(fully_node_id)

        else:
            _ref = fc

        fully_node_id += 1

    elif isinstance(layer, Convolution2D):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        w_data = layer.getWeights().data
        w_orig = w_data
        weight_tensor_name = layer.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(layer.getWeights())
        weight_type_value = type(type_dict[layer.getWeights().dtype](0.0))
        output_type_value = type(type_dict[layer.getOutputTensors()[0].dtype](0.0))


        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])

        arr = w_data.flatten()
        weights_data = ca.getData(np.array(arr).astype(weight_type_value))

        weights_param = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3], w_orig.shape[2], w_orig.shape[1], w_orig.shape[0]),
                                    ca.getOrder("NCHW"), weight_mv_quant_params[0], weight_tensor_name)

        (sY, sX) = layer.getStrideSize()
        (pY, pX) = layer.getPadding()
        dilationFactor = layer.getDilation()
        group = layer.getGroupSize()
        if parser == Parser.Caffe:
            _conv = ca.conv2D_caffe(om, in_, weights_param, sX, sY,
                                    pX[0], pY[0], dilationFactor, group, output_tensor_name)  # Caffe
        elif (parser == Parser.TensorFlowLite or parser == Parser.TensorFlow):
                _conv = ca.conv2D(om, in_, weights_param, sX, sY, pX[0], pX[1], pY[0], pY[1], dilationFactor, group, order_type_dict[output_type_value], mv_quant_params[0], output_tensor_name)

        else:
            throw_error(ErrorTable.ParserNotSupported, parser.name)

        if (output_file is not None):
            if (not(none.match(keep_weights))):
                call_recored_weights(
                    keep_weights, w_orig[:, :, :, ::-1], layer.getWeights().getName().stringifyName())
                if (text.match(keep_weights)):
                    output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) +
                                      '> weightsData' + str(convolution_node_id) +
                                      ' = read_weights_from_file<' + dir_c_type(weight_type_value) +
                                      '>(path + "/projects/Fathom/src2/weights_bias/' +
                                      str(layer.getWeights().getName().stringifyName()) + '.txt");\n')
                elif (binary.match(keep_weights)):
                    output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) +
                                      '> weightsData' + str(convolution_node_id) +
                                      ' = read_weights_from_file<' + dir_c_type(weight_type_value) +
                                      ', ' + weight_buffer_type(weight_type_value) +
                                      '>(path + "/projects/Fathom/src2/weights_bias/' +
                                      str(layer.getWeights().getName().stringifyName()) + '.dat");\n')

            else:
                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) +
                                  '> weightsData' + str(convolution_node_id) + ' = mv::utils::generateSequence<' +
                                  dir_c_type(weight_type_value) + '> ({}*{}*{}*{});\n'.format(w_orig.shape[3], w_orig.shape[2],
                                                                                              w_orig.shape[1], w_orig.shape[0]))

            output_file.write(' ' * 4 + 'auto weights' + str(convolution_node_id) + ' = om.' +
                              constant_type[dir_c_type(weight_type_value)] + '(weightsData' +
                              str(convolution_node_id) + ',{' +
                              str(w_orig.shape[3]) + ',' + str(w_orig.shape[2]) + ',' +
                              str(w_orig.shape[1]) + ',' + str(w_orig.shape[0]) + '}, mv::DType("' +
                              order_type_dict[weight_type_value] + '"), ' + 'mv::Order::getZMajorID(4), ' + '{{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[4])) + '}}, "' +
                              str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto conv' + str(convolution_node_id) + ' = om.conv(' +
                              str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', weights' + str(convolution_node_id) + ', {' +
                              str(sY) + ', ' + str(sX) + '}, {' + str(pX[0]) + ', ' +
                              str(pX[1]) + ', ' + str(pY[0]) + ', ' + str(pY[1]) + '}, ' +
                              str(dilationFactor) + ', ' + str(group) +  ', mv::DType("' +
                              order_type_dict[output_type_value] + '"), '  + '{{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'conv' + \
                str(convolution_node_id)

        if layer.biasEnabled():

            b = layer.getBias()
            bias_tensor_name = layer.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(layer.getBias())[0]
            bias_type_value = type(type_dict[layer.getBias().dtype](0.0))

            bias_data = ca.getData(
                np.array(b.data.flatten()).astype(bias_type_value))

            bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]), ca.getOrder(mcm_1d_layout[parser]), bias_mv_quant_params, bias_tensor_name + "weights")
            _ref = ca.bias(om, _conv, bias, order_type_dict[output_type_value], bias_mv_quant_params, bias_tensor_name)


            if (output_file is not None):

                if (not(none.match(keep_weights))):
                    call_recored_weights(
                        keep_weights, b.data, layer.getBias().getName().stringifyName())
                    if (text.match(keep_weights)):
                        output_file.write(' ' * 4 + 'std::vector<' +
                                          dir_c_type(bias_type_value) +
                                          '> biasWeightsData' +
                                          str(convolution_node_id) +
                                          ' = read_weights_from_file<' +
                                          dir_c_type(bias_type_value) +
                                          '>(path + "/projects/Fathom/src2/weights_bias/' +
                                          str(layer.getBias().getName().stringifyName()) + '.txt");\n')
                    elif (binary.match(keep_weights)):
                        output_file.write(' ' *
                                          4 +
                                          'std::vector<' +
                                          dir_c_type(bias_type_value) +
                                          '> biasWeightsData' +
                                          str(convolution_node_id) +
                                          ' = read_weights_from_file<' +
                                          dir_c_type(bias_type_value) +
                                          ', ' +
                                          bias_buffer_type(bias_type_value) +
                                          '>(path + "/projects/Fathom/src2/weights_bias/' +
                                          str(layer.getBias().getName().stringifyName()) +
                                          '.dat");\n')

                else:
                    output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) + '> biasWeightsData' +
                                      str(convolution_node_id) + ' = mv::utils::generateSequence<' +
                                      dir_c_type(bias_type_value) + '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(convolution_node_id) +
                                  ' = om.' + constant_type[dir_c_type(weight_type_value)] +
                                  '(biasWeightsData' + str(convolution_node_id) +
                                  ',{' + str(b.shape[0]) + '}, mv::DType("' +
                                  order_type_dict[bias_type_value] +
                                  '"), ' + 'mv::Order::getColMajorID(1), ' +
                                  '{{' + ', '.join(map(str, get_parse_quant(layer.getBias())[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[4])) +
                                  '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_c' + str(convolution_node_id) +
                                  ' = om.bias(' + str(tensor_mapping_dict[output_tensor_name]) +
                                  ', biasWeights' + str(convolution_node_id) + ', mv::DType("' +
                                  order_type_dict[bias_type_value] + '"), ' +
                                  '{{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                                  '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_c' + \
                    str(convolution_node_id)

        else:
            _ref = _conv
        convolution_node_id += 1

    elif isinstance(layer, ConvolutionDepthWise2D):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        w_data = layer.getWeights().data
        w_orig = w_data
        weight_tensor_name = layer.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(layer.getWeights())
        weight_type_value = type(type_dict[layer.getWeights().dtype](0.0))
        output_type_value = type(type_dict[layer.getOutputTensors()[0].dtype](0.0))


        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        arr = w_data.flatten()
        weights_data = ca.getData(np.array(arr).astype(weight_type_value))

        weights_param = ca.constant(om, weights_data, ca.getShape(
            w_orig.shape[3], w_orig.shape[2], w_orig.shape[1], w_orig.shape[0]), ca.getOrder("NCHW"), weight_mv_quant_params[0], weight_tensor_name)

        (sY, sX) = layer.getStrideSize()
        (pY, pX) = layer.getPadding()
        dilationFactor = layer.getDilation()
        _conv = ca.depthwiseConv2D(om, in_, weights_param, sX, sY, pX[0], pX[1], pY[0], pY[1], dilationFactor, order_type_dict[output_type_value], mv_quant_params[0], output_tensor_name)  # TFLite


        if (output_file is not None):

            if (keep_weights != "None"):
                call_recored_weights(
                    keep_weights,
                    w_orig,
                    layer.getWeights().getName().stringifyName())
                output_file.write(' ' * 4 + 'std::vector<' +
                                  dir_c_type(weight_type_value) +
                                  '> d_weightsData' +
                                  str(depthwise_node_id) +
                                  ' = read_weights_from_file<' +
                                  dir_c_type(weight_type_value) +
                                  '>(path + "/projects/Fathom/src2/weights_bias/' +
                                  str(layer.getWeights().getName().stringifyName()) +
                                  '.dat");\n')

            else:
                output_file.write(' ' * 4 + 'std::vector<' +
                                  dir_c_type(weight_type_value) +
                                  '> d_weightsData' + str(depthwise_node_id) +
                                  ' = mv::utils::generateSequence<' +
                                  dir_c_type(weight_type_value) +
                                  '> ({}*{}*{}*{});\n'.format(
                                      w_orig.shape[3],
                                      w_orig.shape[2],
                                      w_orig.shape[1],
                                      w_orig.shape[0]))

            output_file.write(' ' * 4 + 'auto d_weights' + str(depthwise_node_id) + ' = om.' + constant_type[dir_c_type(weight_type_value)] +
                              '(d_weightsData' + str(depthwise_node_id) + ',{' + str(w_orig.shape[3]) + ',' +
                              str(w_orig.shape[2]) + ',' + str(w_orig.shape[1]) + ',' + str(w_orig.shape[0]) +
                              '}, mv::DType("' + order_type_dict[weight_type_value] + '"), ' +
                              'mv::Order::getZMajorID(4), ' + '{{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getWeights())[4])) + '}}, "' +
                              str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto depthConv' + str(depthwise_node_id) +
                              ' = om.depthwiseConv(' + str(tensor_mapping_dict[layer.getInputTensors()[0].getName().stringifyName()]) +
                              ', d_weights' + str(depthwise_node_id) + ', {' + str(sY) + ', ' + str(sX) + '}, {' +
                              str(pX[0]) + ', ' + str(pX[1]) + ', ' + str(pY[0]) + ', ' + str(pY[1]) + '}, ' + str(dilationFactor) + ', ' +
                              'mv::DType("' +  order_type_dict[output_type_value] + '"), ' + '{{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) + '},{' +
                              ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}}, "' +
                              str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'depthConv' + \
                str(depthwise_node_id)

        if layer.biasEnabled():

            b = layer.getBias()
            bias_tensor_name = layer.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(layer.getBias())
            bias_type_value = type(type_dict[layer.getBias().dtype](0.0))
            bias_data = ca.getData(
                np.array(b.data.flatten()).astype(bias_type_value))
            bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]), ca.getOrder(
                mcm_1d_layout[parser]), bias_mv_quant_params[0], bias_tensor_name + "weights")
            _ref = ca.bias(
                om, _conv, bias, order_type_dict[output_type_value], mv_quant_params[0], bias_tensor_name)

            if (output_file is not None):
                if (keep_weights != "None"):
                    call_recored_weights(
                        keep_weights, b.data, layer.getBias().getName().stringifyName())
                    output_file.write(' ' * 4 + 'std::vector<' +
                                      dir_c_type(bias_type_value) +
                                      '> biasd_WeightsData' +
                                      str(depthwise_node_id) +
                                      ' = read_weights_from_file<' +
                                      dir_c_type(bias_type_value) +
                                      '>(path + "/projects/Fathom/src2/weights_bias/' +
                                      str(layer.getBias().getName().stringifyName()) + '.dat");\n')

                else:
                    output_file.write(' ' * 4 + 'std::vector<' +
                                      dir_c_type(bias_type_value) +
                                      '> biasd_WeightsData' +
                                      str(depthwise_node_id) +
                                      ' = mv::utils::generateSequence<' +
                                      dir_c_type(bias_type_value) +
                                      '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasdWeights' + str(depthwise_node_id) + ' = om.' +
                                  constant_type[dir_c_type(bias_type_value)] + '(biasd_WeightsData' +
                                  str(depthwise_node_id) + ',{' + str(b.shape[0]) + '}, mv::DType("' +
                                  order_type_dict[bias_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' +
                                  '{{' + ', '.join(map(str, get_parse_quant(layer.getBias())[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getBias())[4])) +
                                  '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_cd' + str(depthwise_node_id) + ' = om.bias(' +
                                  str(tensor_mapping_dict[output_tensor_name]) + ', biasdWeights' +
                                  str(depthwise_node_id) + ', mv::DType("' +
                                  order_type_dict[output_type_value] + '"), ' + '{{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                                  '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) + '}});\n\n')


                tensor_mapping_dict[output_tensor_name] = 'bias_cd' + \
                    str(depthwise_node_id)

        else:
            _ref = _conv
        depthwise_node_id += 1
    # elif isinstance(layer, PReLU):
    #     pred = list(g.predecessors(gnode_name))
    #     in_ = reflist[pred[0]]
    #     slope = layer.getNegativeSlope()
    #     sdata = ca.getData(np.array(slope).astype(np.float64))
    #     s_ = ca.constant(om, sdata, ca.getShape(slope.shape[0]))

    #     _ref = ca.prelu(om, in_, s_)

    elif isinstance(layer, Concat):

        pred = list(g.predecessors(gnode_name))
        # Reorder the pred names based on input tensors order
        input_tensors = layer.getInputTensors()
        ordered_pred = []
        for in_i in input_tensors:
            tensor_name = in_i.getName().stringifyName()
            for l_name in pred:
                l = g.node[l_name]['ref']
                if (l.getOutputTensors()[0].getName().stringifyName() == tensor_name):
                    ordered_pred.append(l.name.stringifyName())
                    break

        print('pred:', pred)
        print('ordered_pred:', ordered_pred)
        in0_ = reflist[ordered_pred[0]]
        output_tensor_name = layer.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(layer.getOutputTensors()[0])
        output_type_value = type_dict[layer.getOutputTensors()[0].dtype]

        vec = ca.pushVector(None, in0_)

        for pair in range(1, len(ordered_pred)):
            in1_ = reflist[ordered_pred[pair]]
            vec = ca.pushVector(vec, in1_)
        in0_ = ca.concat(
            om, vec,order_type_dict[output_type_value], mv_quant_params[0], output_tensor_name)

        if (output_file is not None):

            inp_tensors = []
            for i in layer.getInputTensors():
                inp_tensors.append(i.getName().stringifyName())

            mcm_inputs = []
            for i in inp_tensors:
                mcm_inputs.append(tensor_mapping_dict[str(i)])

            output_file.write(' ' * 4 + 'auto concat' + str(concat_node_id) + ' = om.concat({' +
                              ', '.join(map(str, mcm_inputs)) + '}, "C", ' + 'mv::DType("' + order_type_dict[output_type_value] + '"), ' +
                              '{{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[1])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[2])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[3])) +
                              '},{' + ', '.join(map(str, get_parse_quant(layer.getOutputTensors()[0])[4])) +
                              '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'concat' + \
                str(concat_node_id)

        _ref = in0_
        concat_node_id += 1

    assert _ref is not None, "layer unsupported for C++ Conversion" + \
        str(type(layer))
    return _ref, om
