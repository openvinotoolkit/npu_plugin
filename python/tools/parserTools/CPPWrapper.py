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

import sys
import os
base = os.environ.get('MCM_HOME')
ldlib = os.environ.get('LD_LIBRARY_PATH')
if base is None:
    print("Please set environment path MCM_HOME. Exiting...")
    quit()
if ldlib is None:
    print("Please set your LD_LIBRARY_PATH environment variable correctly. Exiting...")
    quit()
sys.path.append(base + '/python/api')
import composition_api as ca
from parserTools.GraphUtils import buildGraph
from parserTools.EnumController import throw_error
import networkx as nx
import numpy as np
from parserTools.Models.EnumDeclarations import Parser, ErrorTable

convolution_node_id = 0
pooling_node_id = 0
relu_node_id = 0
dropout_node_id = 0
softmax_node_id = 0
eltwise_node_id = 0
concat_node_id = 0
depthwise_node_id = 0
scale_node_id = 0

def initialize_execution_file(comp_un, op_mod):

    exec_file = open("mcm_network.cpp", "w+")
    exec_file.write('//This file is the parsed network which is created through python.\n')
    exec_file.write('#include ' + '"' +'include/mcm/compiler/compilation_unit.hpp' + '"\n')
    exec_file.write('#include ' + '"' +'include/mcm/utils/data_generator.hpp' + '"\n')
    exec_file.write('#include ' + '"' +'include/mcm/utils/serializer/Fp16Convert.h' + '"\n')
    exec_file.write('#include ' + '"' +'meta/include/mcm/op_model.hpp' + '"\n')
    exec_file.write('#include ' + '"' +'include/mcm/utils/hardware_tests.hpp' + '"\n\n')

    exec_file.write('#include ' + '"' +'iostream' + '"\n')
    exec_file.write('#include ' + '"' +'fstream' + '"\n\n')

    exec_file.write('int main()\n{\n')
    exec_file.write(' ' * 4 + 'double inf = std::numeric_limits<double>::infinity();\n\n')

    exec_file.write(' ' * 4 + 'mv::CompilationUnit unit("testModel");\n')
    exec_file.write(' ' * 4 + 'mv::OpModel& om = unit.model();\n')
    return exec_file

def finalize_execution_file(exec_file):

    exec_file.write(' ' * 4 + 'std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";\n')
    exec_file.write(' ' * 4 + 'unit.loadCompilationDescriptor(compDescPath);\n\n')

    exec_file.write(' ' * 4 + 'unit.loadTargetDescriptor(mv::Target::ma2490);\n')
    exec_file.write(' ' * 4 + 'unit.initialize();\n')
    exec_file.write(' ' * 4 + 'unit.run();\n\n')

    exec_file.write(' ' * 4 + 'system("dot -Tpng original_model.dot -o original_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng adapt_model.dot -o adapt_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng keembay_adapt_model.dot -o keembay_adapt_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng dma_model.dot -o dma_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng final_model.dot -o final_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng TransitiveReduction.dot -o TransitiveReduction.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng deallocation_model_data.dot -o deallocation_model_data.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng DmaControlFlows_model.dot -o DmaControlFlows_model.png");\n')
    exec_file.write(' ' * 4 + 'system("dot -Tpng InputOutputControlFlows_model.dot -o InputOutputControlFlows_model.png");\n')
    exec_file.write(' ' * 4 + 'system("flatc -t ../../schema/graphfile/src/schema/graphfile.fbs -- blob.bin");\n}\n')
    return exec_file

def ComposeForCpp(parsedLayers, arguments):

#WRAPPER AND COMPOSITION INTERFACE SUPPORT MYRIAD X AS WELL
#ARGS (X, KMB) AND CAFFE PARSER IS INSERTED AS WELL

    reference_list = {}
    target_desc = "ma2490"

    comp_unit = ca.getCompilationUnit(target_desc)

    # if user provided compilation descriptor file
    if arguments.comp_descriptor is not None:
        ca.loadCompilationDescriptor(comp_unit, arguments.comp_descriptor)

    # return the object model
    om = ca.getModel(comp_unit)

    g = buildGraph(parsedLayers)

    if (arguments.produceMcmDescriptor == True):
        mcm_file = initialize_execution_file(comp_unit, om)
    else:
        mcm_file = None

    tensor_mapping_dict = {}
    for index, child in enumerate(nx.lexicographical_topological_sort(g)):
        layer = g.node[child]['ref']
        n = layer.name.stringifyName()
        print ("Layer {} is parsed".format(n))
        reference_list[n], om = buildOM(g, child, reference_list, om, True, Parser.TensorFlowLite, mcm_file, tensor_mapping_dict)

    if (arguments.produceMcmDescriptor == True):
        finalize_execution_file(mcm_file)

    exit()
    print("Compiling...")
    ca.compile(comp_unit)
    ca.deleteCompilationUnitObject(comp_unit)

def call_the_appropriate_wrap(platform, type, om, shape, type_value, quant_params, order, output_tensor_name):

    wrapper_dict = {
                'Input': ca.input
                }

    if (platform == 'KMB'):
        return wrapper_dict[type](om, shape, type_value, quant_params, order, output_tensor_name)
    else:
        return wrapper_dict[type](om, shape, type_value, order)

def get_parse_quant(tensor):

    try:
        tensor_quant = tensor.getQuantizationParameters()
    except AttributeError:
        return ca.getQuantParams(ca.getData(np.array([], dtype=np.int64)), ca.getData(np.array([], dtype=np.float64)),
            ca.getData(np.array([], dtype=np.float64)), ca.getData(np.array([], dtype=np.float64)))

    try:
        zero_quant_data = tensor_quant.getZeroPoint().astype(np.int64)
        zero_data = ca.getData(zero_quant_data)
    except ValueError:
        zero_quant_data = np.array([], dtype=np.int64)
        zero_data = ca.getData(np.array([], dtype=np.int64))

    try:
        scale_quant_data = tensor_quant.getScale().astype(np.float64)
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

    mv_quant_params = ca.getQuantParams(zero_data, scale_data, min_quant_data, max_quant_data)

    return mv_quant_params, zero_quant_data, scale_quant_data, min_quant_param, max_quant_param

def dir_c_type(type_value):

    if isinstance(type_value(), int):
        return 'int64_t'
    else:
        return 'double'

def buildOM(g, gnode_name, reflist, om, kmb, parser, output_file = None, tensor_mapping_dict = {}):
    # """
    #     Construct C++ Representation of a Layer.
    #     Return the iterator for passing forward
    # """

    gnode = g.node[gnode_name]
    l = gnode['ref']

    from parserTools.Parser.InnerProduct import InnerProduct
    from parserTools.Parser.Input import Input
    from parserTools.Parser.Output import Output
    from parserTools.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
    from parserTools.Parser.Pooling import Pooling
    from parserTools.Parser.Eltwise import Eltwise
    from parserTools.Parser.ReLU import ReLU
    from parserTools.Parser.PReLU import PReLU
    from parserTools.Parser.Concat import Concat
    from parserTools.Parser.Scale import Scale
    from parserTools.Parser.Bias import Bias
    from parserTools.Parser.BatchNorm import BatchNorm
    from parserTools.Parser.Softmax import Softmax
    from parserTools.Parser.NoOp import NoOp

    _ref = None

    type_dict = {
                np.uint8: np.int,
                np.dtype('uint8'): np.int,
                np.dtype('int32'): np.int,
                np.float16: np.float64
                }

    order_type_dict = {
                       np.int: "UInt8",
                      }

    platform_dict = {
                True: 'KMB',
                False: 'my-X'
                }

    mcm_4d_layout = {
            Parser.Caffe: "NCHW",
            Parser.TensorFlowLite: "NHWC"
    }

    mcm_1d_layout = {
            Parser.Caffe: "W",
            Parser.TensorFlowLite: "W"
    }

    mcm_2d_layout = {
        Parser.Caffe: "HW",
        Parser.TensorFlowLite: "WC"
    }

    constant_type = {
        'int64_t': 'constantInt',
        'double': 'constant'
    }

    global convolution_node_id, pooling_node_id, relu_node_id, dropout_node_id, eltwise_node_id, softmax_node_id, depthwise_node_id, concat_node_id, scale_node_id
    platform = platform_dict[kmb]

    if isinstance(l, Bias):
        assert 0, "Standalone Bias Layer currently unsupported by C++ API"

    if isinstance(l, Input):

        s = l.getOutputTensors()[0].getShape() #getShape returns N, C, H, W
        shape = ca.getShape(s[3], s[2], s[1], s[0])
        type_value = type_dict[l.getOutputTensors()[0].dtype](0.0)
        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        _ref = call_the_appropriate_wrap(platform_dict[kmb], 'Input', om, shape, type_value, ca.getOrder(mcm_4d_layout[parser]),  \
            get_parse_quant(l.getOutputTensors()[0])[0], output_tensor_name)
        
        if (output_file != None):

            output_file.write(' ' * 4 + 'auto input0 = om.input({' + str(s[3]) + ',' + str(s[2]) + ',' + str(s[1]) + ',' + str(s[0]) + '}, mv::DType("' + \
                order_type_dict[type(type_value)] + '"), ' + 'mv::Order::getZMajorID(4), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name) + '");\n\n')
            tensor_mapping_dict[output_tensor_name] = 'input0'

    elif isinstance(l, Output):

        input_tensor = tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]
        pred = list(g.predecessors(gnode_name))
        if (platform == 'KMB'):
            _ref = ca.output(om, reflist[pred[0]], '')
        else:
            _ref = ca.output(om, reflist[pred[0]])

        if (output_file != None):
            output_file.write(' ' * 4 + 'om.output(' + str(input_tensor) + ');\n\n')

    elif isinstance(l, Pooling):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        ry, rx = l.getKernelSize()
        sy, sx = l.getStride()
        py, px = l.getPadding()
        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()

        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]

        if l.getType() == Pooling.Type.MAX:

            if (parser == Parser.Caffe):
                if (platform == 'KMB'):
                    _ref = ca.maxpool2D_caffe(om, in_, rx, ry, sx, sy, px[0], py[0], output_tensor_name)
                else:
                    _ref = ca.maxpool2D_caffe(om, in_, rx, ry, sx, sy, px[0], py[0])
            elif parser == Parser.TensorFlowLite:
                _ref = ca.maxpool2D(om, in_, rx, ry, sx, sy, px[0], py[0], mv_quant_params, output_tensor_name)
            else:
                throw_error(ErrorTable.ParserNotSupported, parser.name)

        elif l.getType() == Pooling.Type.AVE:

            if (parser == Parser.Caffe):
                if (platform == 'KMB'):
                    _ref = ca.avgpool2D_caffe(om, in_, rx, ry, sx, sy, px[0], py[0], output_tensor_name)
                else:
                    _ref = ca.avgpool2D_caffe(om, in_, rx, ry, sx, sy, px[0], py[0])
            elif parser == Parser.TensorFlowLite:
                _ref = ca.avgpool2D(om, in_, rx, ry, sx, sy, px[0], py[0], mv_quant_params, output_tensor_name)
            else:
                throw_error(ErrorTable.ParserNotSupported, parser.name)

        if (output_file != None):

            pool_mcm = {
                Pooling.Type.MAX:'maxPool',
                Pooling.Type.AVE:'averagePool'
            }

            output_file.write(' ' * 4 + 'auto pool' + str(pooling_node_id) + ' = om.' + str(pool_mcm[l.getType()]) + '(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + \
                ', {' + str(ry) + ', ' + str(rx) + '}, {' + str(sy) + ', ' + str(sx) + '}, {' + str(px[0]) + ', ' + str(px[1]) + ', ' + str(py[0]) + ', ' + str(py[1]) + '}, ' + \
                'true, "", "floor", {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name)  + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'pool' + str(pooling_node_id)
            pooling_node_id += 1

    elif isinstance(l, ReLU):

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        if (platform == 'KMB'):
            _ref = ca.relu(om, in_, mv_quant_params, output_tensor_name)
        else:
            _ref = ca.relu(om, in_)

        if (output_file != None):

            output_file.write(' ' * 4 + 'auto relu' + str(relu_node_id) + ' = om.relu(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + \
                ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name)  + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'relu' + str(relu_node_id)
            relu_node_id += 1

    elif isinstance(l, NoOp):

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        if (platform == 'KMB'):
            _ref = ca.dropOut(om, in_, mv_quant_params, output_tensor_name)
        else:
            _ref = ca.dropOut(om, in_)

        if (output_file != None):

            output_file.write(' ' * 4 + 'auto dropout' + str(dropout_node_id) + ' = om.dropout(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + \
                ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name)  + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'dropout' + str(dropout_node_id)
            dropout_node_id += 1

    elif isinstance(l, Eltwise):

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        pred = list(g.predecessors(gnode_name))
        in1_ = reflist[pred[0]]

        if l.getType() == Eltwise.Type.WSUM:

            if len(pred) == 1:
                in2_ = reflist[pred[0]]  # Same input.
            else:
                in2_ = reflist[pred[1]]

            if (platform == 'KMB'):
                _ref = ca.add(om, in1_, in2_, mv_quant_params, output_tensor_name)
            else:
                _ref = ca.add(om, in1_, in2_)

        elif l.getType() == Eltwise.Type.WPROD:

            in2_ = reflist[pred[1]]

            if (platform == 'KMB'):
                _ref = ca.multiply(om, in1_, in2_, mv_quant_params, output_tensor_name)
            else:
                _ref = ca.multiply(om, in1_, in2_)

        if (output_file != None):

            eltwise_map = {
                Eltwise.Type.WSUM:'add',
                Eltwise.Type.WPROD:'multiply',
            }

            output_file.write(' ' * 4 + 'auto eltwise' + str(eltwise_node_id) + ' = om.' + str(eltwise_map[l.getType()]) + '(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + \
                ',' + str(tensor_mapping_dict[l.getInputTensors()[1].getName().stringifyName()]) + ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name)  + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'eltwise' + str(eltwise_node_id)
            eltwise_node_id += 1

    elif isinstance(l, Scale) or isinstance(l, BatchNorm):
      
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        #NOT SURE IF THE MULTIPLIER IS 1DIM
        scale_data = l.getMultiplier()
        scale_vector = ca.getData(scale_data.astype(type_dict[l.getMultiplier().dtype]))
        scale_tensor_name = l.getMultiplier().getName().stringifyName()
        scale_mv_quant_params = get_parse_quant(l.getMultiplier())[0]
        scale_type_value = type(type_dict[l.getMultiplier().dtype](0.0))

        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()

        if (platform == 'KMB'):
            scale_param = ca.constant(om, scale_vector, ca.getShape(scale_data.shape[0]), ca.getOrder(mcm_1d_layout[parser]), scale_mv_quant_params, scale_tensor_name)
            scale = ca.scale(om, in_, scale_param, mv_quant_params, output_tensor_name)
        else:
            scale_param = ca.constant(om, scale_vector, ca.getShape(scale_data.shape[0]))
            scale = ca.scale(om, in_, scale_param)


        if (output_file != None):

            output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(scale_type_value) + '> weightsData' + str(scale_node_id) + ' = mv::utils::generateSequence<' + \
                dir_c_type(scale_type_value) + '> ({}*{}*{}*{});\n'.format(scale_data.shape[0]))

            output_file.write(' ' * 4 + 'auto weights' + str(scale_node_id) + ' = om.'+ constant_type[dir_c_type(scale_type_value)] + '(weightsData' + str(scale_node_id) +',{' +\
                str(scale_data.shape[0]) + '}, mv::DType("' + \
                order_type_dict[scale_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getMultiplier())[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getMultiplier())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getMultiplier())[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getMultiplier())[4])) + '}}, "' + str(scale_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto scale' + str(scale_node_id) + ' = om.scale(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + ', weights' + str(scale_node_id) + \
                ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'scale' + str(scale_node_id)

        if l.hasBiasBeta():

            bias_data = l.getBiasBeta()
            bias_vector = ca.getData(bias_data.astype(type_dict[l.getBias().dtype]))
            bias_tensor_name = l.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(l.getBias())
            bias_type_value = type(type_dict[l.getBias().dtype](0.0))

            if (platform == 'KMB'):
                bias_param = ca.constant(om, bias_vector, ca.getShape(bias_data.shape[0]), ca.getOrder(mcm_1d_layout[parser]), bias_quant_params, bias_tensor_name)
                bias = ca.bias(om, scale, bias_param, bias_quant_params, bias_tensor_name)
            else:
                bias_param = ca.constant(om, bias_vector, ca.getShape(bias_data.shape[0]))
                bias = ca.bias(om, scale, bias_param)

            if (output_file != None):

                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) + '> biasWeightsData' + str(scale_node_id) + ' = mv::utils::generateSequence<' + \
                    dir_c_type(bias_type_value) + '> ({});\n'.format(bias_data.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(scale_node_id) + ' = om.'+ constant_type[dir_c_type(bias_type_value)] + '(biasWeightsData' + str(scale_node_id) +',{' +\
                    str(bias_data.shape[0]) + '}, mv::DType("' + \
                    order_type_dict[bias_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getBiasBeta())[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getBiasBeta())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getBiasBeta())[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getBiasBeta())[4])) + '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_s' + str(scale_node_id) + ' = om.bias(' + str(tensor_mapping_dict[output_tensor_name]) + ', biasWeights' + str(scale_node_id) + \
                    ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_i' + str(scale_node_id)
                scale_node_id += scale_node_id

            _ref = bias
        else:
            _ref = scale

    # elif isinstance(l, BatchNorm):

    #     pred = list(g.predecessors(gnode_name))
    #     in_ = reflist[pred[0]]

    #     '''mean_data = ca.getData(l.getMean().astype(np.float64))
    #     mean_param = ca.constant(om, mean_data, ca.getShape(*l.getMean().shape))

    #     var_data = ca.getData(l.getVariance().astype(np.float64))
    #     variance_param = ca.constant(om, var_data, ca.getShape(*l.getVariance().shape))

    #     offset_data = ca.getData(l.getBiasBeta().astype(np.float64))
    #     offset_param = ca.constant(om, offset_data, ca.getShape(*l.getBiasBeta().shape))

    #     scale_data = ca.getData(l.getMultiplier().astype(np.float64))
    #     scale_param = ca.constant(om, scale_data, ca.getShape(*l.getMultiplier().shape))

    #     eps = l.getEPS()

    #     _ref = ca.batchNorm(om, in_, mean_param, variance_param, offset_param, scale_param, eps)'''

    #     scale_data = l.getMultiplier()
    #     scale_vector = ca.getData(scale_data.astype(np.float64))
    #     scale_param = ca.constant(om, scale_vector, ca.getShape(scale_data.shape[0]))
    #     scale = ca.scale(om, in_, scale_param)

    #     bias_data = l.getBiasBeta()
    #     bias_vector = ca.getData(bias_data.astype(np.float64))
    #     bias_param = ca.constant(om, bias_vector, ca.getShape(bias_data.shape[0]))
    #     bias = ca.bias(om, scale, bias_param)
    #     _ref = bias

    elif isinstance(l, Softmax):

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]

        if (platform == 'KMB'):
            _ref = ca.softmax(om, in_, mv_quant_params, output_tensor_name)
        else:
            _ref = ca.softmax(om, in_)

        if (output_file != None):

            output_file.write(' ' * 4 + 'auto softmax' + str(softmax_node_id) + ' = om.softmax(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + \
                ', "C", {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name)  + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'softmax' + str(softmax_node_id)
            softmax_node_id += 1

    elif isinstance(l, InnerProduct):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        w_data = l.getWeights().data
        w_orig = w_data

        w_data = np.transpose(w_data, (3, 2, 1, 0))
        arr = w_data.flatten()
        weight_tensor_name = l.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(l.getWeights())
        weight_type_value = type(type_dict[l.getWeights().dtype](0.0))
        weight_data = ca.getData(np.array(arr).astype(weight_type_value))

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]

        if (platform == 'KMB'):
            weights_ = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3], w_orig.shape[2]), ca.getOrder(mcm_2d_layout[parser]), weight_mv_quant_params, weight_tensor_name)
            fc = ca.fullyConnected(om, in_, weights_, mv_quant_params, output_tensor_name)
        else:
            weights_ = ca.constant(om, weight_data, ca.getShape(w_orig.shape[3], w_orig.shape[2]))
            fc = ca.fullyConnected(om, in_, weights_)

        if (output_file != None):

            output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) + '> weightsData' + str(fully_node_id) + ' = mv::utils::generateSequence<' + \
                dir_c_type(weight_type_value) + '> ({}*{}*{}*{});\n'.format(w_orig.shape[3], w_orig.shape[2]))

            output_file.write(' ' * 4 + 'auto weights' + str(fully_node_id) + ' = om.'+ constant_type[dir_c_type(weight_type_value)] + '(weightsData' + str(fully_node_id) +',{' +\
                str(w_orig.shape[3]) + ',' + str(w_orig.shape[2])+ '}, mv::DType("' + \
                order_type_dict[weight_type_value] + '"), ' + 'mv::Order::getZMajorID(2), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getWeights())[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getWeights())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getWeights())[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getWeights())[4])) + '}}, "' + str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto fc' + str(fully_node_id) + ' = om.fullyConnected(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + ', weights' + str(fully_node_id) + \
                ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'fc' + str(fully_node_id)

        if l.biasEnabled():

            b = l.getBias()
            bias_tensor_name = l.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(l.getBias())
            bias_data = ca.getData(np.array(b.data.flatten()).astype(bias_type_value))
            bias_type_value = type(type_dict[l.getBias().dtype](0.0))

            if (platform == 'KMB'):
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[2]), ca.getOrder(mcm_1d_layout[parser]), bias_mv_quant_params, bias_tensor_name+"weights")
                _ref = ca.bias(om, fc, bias, mv_quant_params, bias_tensor_name)
            else:
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[2]))
                _ref = ca.bias(om, fc, bias)

            if (output_file != None):

                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) + '> biasWeightsData' + str(fully_node_id) + ' = mv::utils::generateSequence<' + \
                    dir_c_type(bias_type_value) + '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(fully_node_id) + ' = om.'+ constant_type[dir_c_type(bias_type_value)] + '(biasWeightsData' + str(fully_node_id) +',{' +\
                    str(b.shape[0]) + '}, mv::DType("' + \
                    order_type_dict[bias_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getBias())[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getBias())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getBias())[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getBias())[4])) + '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_i' + str(fully_node_id) + ' = om.bias(' + str(tensor_mapping_dict[output_tensor_name]) + ', biasWeights' + str(fully_node_id) + \
                    ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_i' + str(fully_node_id)

        else:
            _ref = fc

        fully_node_id += 1

    elif isinstance(l, Convolution2D):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        w_data = l.getWeights().data
        w_orig = w_data
        weight_tensor_name = l.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(l.getWeights())[0]
        weight_type_value = type(type_dict[l.getWeights().dtype](0.0))

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]

        arr = w_data.flatten()
        weights_data = ca.getData(np.array(arr).astype(weight_type_value))
        if (platform == 'KMB'):
            weights_param = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3],
                                                                      w_orig.shape[2],
                                                                      w_orig.shape[1],
                                                                      w_orig.shape[0]), ca.getOrder("NCHW"), weight_mv_quant_params, weight_tensor_name)
        else:
            weights_param = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3],
                                                                      w_orig.shape[2],
                                                                      w_orig.shape[1],
                                                                      w_orig.shape[0]))
        (sY, sX) = l.getStrideSize()
        (pY, pX) = l.getPadding()
        dilationFactor = l.getDilation()
        group = l.getGroupSize()

        if parser == Parser.Caffe:
            if (platform == 'KMB'):
                _conv = ca.conv2D_caffe(om, in_, weights_param, sX, sY, pX[0], pY[0], dilationFactor, group, output_tensor_name)  # Caffe
            else:
                _conv = ca.conv2D_caffe(om, in_, weights_param, sX, sY, pX[0], pY[0], dilationFactor, group)  # Caffe
        elif parser == Parser.TensorFlowLite:
            _conv = ca.conv2D(om, in_, weights_param, sX, sY, pX[0], pY[0], dilationFactor, group, mv_quant_params, output_tensor_name)  # TFLite
        else:
            throw_error(ErrorTable.ParserNotSupported, parser.name)

        if (output_file != None):

            output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) + '> weightsData' + str(convolution_node_id) + ' = mv::utils::generateSequence<' + \
                dir_c_type(weight_type_value) + '> ({}*{}*{}*{});\n'.format(w_orig.shape[3], w_orig.shape[2], w_orig.shape[1],w_orig.shape[0]))

            output_file.write(' ' * 4 + 'auto weights' + str(convolution_node_id) + ' = om.'+ constant_type[dir_c_type(weight_type_value)] + '(weightsData' + str(convolution_node_id) +',{' +\
                str(w_orig.shape[3]) + ',' + str(w_orig.shape[2]) + ',' + str(w_orig.shape[1]) + ',' + str(w_orig.shape[0]) + '}, mv::DType("' + \
                order_type_dict[weight_type_value] + '"), ' + 'mv::Order::getZMajorID(4), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getWeights())[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getWeights())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getWeights())[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getWeights())[4])) + '}}, "' + str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto conv' + str(convolution_node_id) + ' = om.conv(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + ', weights' + str(convolution_node_id) + \
                ', {' + str(sY) + ', ' + str(sX) + '}, {' + str(pX[0]) + ', ' + str(pX[0]) + ', ' + str(pY[0]) + ', ' + str(pY[0]) + '}, ' + \
                str(dilationFactor) + ', ' + str(group) + ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'conv' + str(convolution_node_id)

        if l.biasEnabled():

            b = l.getBias()
            bias_tensor_name = l.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(l.getBias())[0]

            bias_type_value = type(type_dict[l.getBias().dtype](0.0))
            bias_data = ca.getData(np.array(b.data.flatten()).astype(bias_type_value))

            if (platform == 'KMB'):
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]), ca.getOrder(mcm_1d_layout[parser]), bias_mv_quant_params, bias_tensor_name+"weights")
                _ref = ca.bias(om, _conv, bias, mv_quant_params, bias_tensor_name)
            else:
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]))
                _ref = ca.bias(om, bias_data, ca.getShape(b.shape[0]))

            if (output_file != None):

                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) + '> biasWeightsData' + str(convolution_node_id) + ' = mv::utils::generateSequence<' + \
                    dir_c_type(bias_type_value) + '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(convolution_node_id) + ' = om.'+ constant_type[dir_c_type(weight_type_value)] + '(biasWeightsData' + str(convolution_node_id) +',{' +\
                    str(b.shape[0]) + '}, mv::DType("' + \
                    order_type_dict[bias_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getBias())[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getBias())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getBias())[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getBias())[4])) + '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_c' + str(convolution_node_id) + ' = om.bias(' + str(tensor_mapping_dict[output_tensor_name]) + ', biasWeights' + str(convolution_node_id) + \
                    ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_c' + str(convolution_node_id)
        else:
            _ref = _conv

        convolution_node_id += 1

    elif isinstance(l, ConvolutionDepthWise2D):

        pred = list(g.predecessors(gnode_name))
        in_ = reflist[pred[0]]
        w_data = l.getWeights().data
        w_orig = w_data
        weight_tensor_name = l.getWeights().getName().stringifyName()
        weight_mv_quant_params = get_parse_quant(l.getWeights())[0]
        weight_type_value = type(type_dict[l.getWeights().dtype](0.0))

        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]

        arr = w_data.flatten()
        weights_data = ca.getData(np.array(arr).astype(weight_type_value))
        if (platform == 'KMB'):
            weights_param = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3],
                                                                      w_orig.shape[2],
                                                                      w_orig.shape[1],
                                                                      w_orig.shape[0]), ca.getOrder("NCHW"), weight_mv_quant_params, weight_tensor_name)
        else:
            weights_param = ca.constant(om, weights_data, ca.getShape(w_orig.shape[3],
                                                                      w_orig.shape[2],
                                                                      w_orig.shape[1],
                                                                      w_orig.shape[0]))
        (sY, sX) = l.getStrideSize()
        (pY, pX) = l.getPadding()
        dilationFactor = l.getDilation()

        _conv = ca.depthwiseConv2D(om, in_, weights_param, sX, sY, dilationFactor, pX[0], pY[0], mv_quant_params, output_tensor_name)  # TFLite

        if (output_file != None):

            output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(weight_type_value) + '> weightsData' + str(depthwise_node_id) + ' = mv::utils::generateSequence<' + \
                dir_c_type(weight_type_value) + '> ({}*{}*{}*{});\n'.format(w_orig.shape[3], w_orig.shape[2], w_orig.shape[1],w_orig.shape[0]))

            output_file.write(' ' * 4 + 'auto weights' + str(depthwise_node_id) + ' = om.'+ constant_type[dir_c_type(weight_type_value)] + '(weightsData' + str(depthwise_node_id) +',{' +\
                str(w_orig.shape[3]) + ',' + str(w_orig.shape[2]) + ',' + str(w_orig.shape[1]) + ',' + str(w_orig.shape[0]) + '}, mv::DType("' + \
                order_type_dict[weight_type_value] + '"), ' + 'mv::Order::getZMajorID(4), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getWeights())[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getWeights())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getWeights())[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getWeights())[4])) + '}}, "' + str(weight_tensor_name) + '");\n')

            output_file.write(' ' * 4 + 'auto depthConv' + str(depthwise_node_id) + ' = om.conv(' + str(tensor_mapping_dict[l.getInputTensors()[0].getName().stringifyName()]) + ', weights' + str(convolution_node_id) + \
                ', {' + str(sY) + ', ' + str(sX) + '}, {' + str(pX[0]) + ', ' + str(pX[0]) + ', ' + str(pY[0]) + ', ' + str(pY[0]) + '}, ' + \
                str(dilationFactor) + ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'depthConv' + str(depthwise_node_id)

        if l.biasEnabled():

            b = l.getBias()
            bias_tensor_name = l.getBias().getName().stringifyName()
            bias_mv_quant_params = get_parse_quant(l.getBias())[0]

            bias_type_value = type(type_dict[l.getBias().dtype](0.0))
            bias_data = ca.getData(np.array(b.data.flatten()).astype(bias_type_value))

            if (platform == 'KMB'):
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]), ca.getOrder(mcm_1d_layout[parser]), bias_mv_quant_params, bias_tensor_name+"weights")
                _ref = ca.bias(om, _conv, bias, mv_quant_params, bias_tensor_name)
            else:
                bias = ca.constant(om, bias_data, ca.getShape(b.shape[0]))
                _ref = ca.bias(om, bias_data, ca.getShape(b.shape[0]))

            if (output_file != None):

                output_file.write(' ' * 4 + 'std::vector<' + dir_c_type(bias_type_value) + '> biasWeightsData' + str(depthwise_node_id) + ' = mv::utils::generateSequence<' + \
                    dir_c_type(bias_type_value) + '> ({});\n'.format(b.shape[0]))

                output_file.write(' ' * 4 + 'auto biasWeights' + str(depthwise_node_id) + ' = om.'+ constant_type[dir_c_type(weight_type_value)] + '(biasWeightsData' + str(depthwise_node_id) +',{' +\
                    str(b.shape[0]) + '}, mv::DType("' + \
                    order_type_dict[bias_type_value] + '"), ' + 'mv::Order::getColMajorID(1), ' + '{{' + ', '.join(map(str, get_parse_quant(l.getBias())[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getBias())[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getBias())[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getBias())[4])) + '}}, "' + str(bias_tensor_name) + '");\n')

                output_file.write(' ' * 4 + 'auto bias_cd' + str(depthwise_node_id) + ' = om.bias(' + str(tensor_mapping_dict[output_tensor_name]) + ', biasWeights' + str(convolution_node_id) + \
                    ', {{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                    '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                    ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}});\n\n')

                tensor_mapping_dict[output_tensor_name] = 'bias_cd' + str(depthwise_node_id)
        else:
            _ref = _conv

        depthwise_node_id += 1

    # elif isinstance(l, PReLU):
    #     pred = list(g.predecessors(gnode_name))
    #     in_ = reflist[pred[0]]
    #     slope = l.getNegativeSlope()
    #     sdata = ca.getData(np.array(slope).astype(np.float64))
    #     s_ = ca.constant(om, sdata, ca.getShape(slope.shape[0]))

    #     _ref = ca.prelu(om, in_, s_)

    elif isinstance(l, Concat):

        inp_tensors = []
        for i in l.getInputTensors():
            inp_tensors.append(i.getName().stringifyName())

        pred = list(g.predecessors(gnode_name))

        in0_ = reflist[pred[0]]
        output_tensor_name = l.getOutputTensors()[0].getName().stringifyName()
        mv_quant_params = get_parse_quant(l.getOutputTensors()[0])[0]
        vec = ca.pushVector(None, in0_)

        for pair in range(1, len(pred)):
            in1_ = reflist[pred[pair]]
            vec = ca.pushVector(vec, in1_)

        in0_ = ca.concat(om, vec)

        if (output_file != None):

            mcm_inputs = []
            for i in inp_tensors:
                mcm_inputs.append(tensor_mapping_dict[str(i)])

            output_file.write(' ' * 4 + 'auto concat' + str(concat_node_id) + ' = om.concat({' + ', '.join(map(str, mcm_inputs)) + '}, "C", ' +  \
                '{{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[1])) + \
                '},{' + ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[2]))  + '},{' +', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[3])) + '},{' + \
                ', '.join(map(str, get_parse_quant(l.getOutputTensors()[0])[4])) + '}}, "' + str(output_tensor_name) + '");\n\n')

            tensor_mapping_dict[output_tensor_name] = 'concat' + str(concat_node_id)

        _ref = in0_
        concat_node_id +=1

    else:
        print("NOT FOUND")

    assert _ref is not None, "layer unsupported for C++ Conversion" + str(type(l))
    return _ref, om