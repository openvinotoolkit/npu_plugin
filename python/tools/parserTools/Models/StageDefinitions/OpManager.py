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

from Models.EnumDeclarations import *
from Controllers.EnumController import *

from Models.StageDefinitions.AveragePooling import *
from Models.StageDefinitions.Bias import *
from Models.StageDefinitions.StorageOrderConvert import *
from Models.StageDefinitions.Convolution import *
from Models.StageDefinitions.Copy import *
from Models.StageDefinitions.Crop import *
from Models.StageDefinitions.Deconv import *
from Models.StageDefinitions.DepthConv import *
from Models.StageDefinitions.Eltwise import *
from Models.StageDefinitions.Elu import *
from Models.StageDefinitions.FCL import *
from Models.StageDefinitions.MyriadXHardwareLayer import *
from Models.StageDefinitions.InnerLRN import *
from Models.StageDefinitions.LRN import *
from Models.StageDefinitions.MaxConst import *
from Models.StageDefinitions.MaxPooling import *
from Models.StageDefinitions.NoOp import *
from Models.StageDefinitions.Prelu import *
from Models.StageDefinitions.Power import *
from Models.StageDefinitions.Relu import *
from Models.StageDefinitions.Reshape import *
from Models.StageDefinitions.Rsqrt import *
from Models.StageDefinitions.Scale import *
from Models.StageDefinitions.ScaleScalar import *
from Models.StageDefinitions.Softmax import *
from Models.StageDefinitions.Square import *
from Models.StageDefinitions.Sigmoid import *
from Models.StageDefinitions.SumReduce import *
from Models.StageDefinitions.TanH import *
from Models.StageDefinitions.ToPlaneMajor import *
from Models.StageDefinitions.Permute import *
from Models.StageDefinitions.Normalize import *
from Models.StageDefinitions.PriorBox import *
from Models.StageDefinitions.DetectionOutput import *
from Models.StageDefinitions.Relu_Op import ReluOp
from Models.StageDefinitions.Elu_Op import EluOp
from Models.StageDefinitions.Upsampling import Upsampling

def get_op_definition(op_type, force_op=False):
    """
    Get the global definition of an operation
    :param op_type: the operation to lookup
    :return: definition object
    """

    op_mapping = {
        StageType.average_pooling:                  AveragePooling(),
        StageType.bias:                             Bias(),
        StageType.storage_order_convert:            StorageOrderConvert(),
        StageType.convolution:                      Convolution(),
        StageType.copy:                             Copy(),
        StageType.crop:                             Crop(),
        StageType.deconvolution:                    Deconv(),
        StageType.depthwise_convolution:            DepthConv(),
        StageType.eltwise_prod:                     Eltwise(),
        StageType.eltwise_sum:                      Eltwise(),
        StageType.eltwise_max:                      Eltwise(),
        StageType.elu:                              Elu(),
        StageType.fully_connected_layer:            FCL(),
        StageType.innerlrn:                         InnerLRN(),
        StageType.LRN:                              LRN(),
        StageType.max_pooling:                      MaxPooling(),
        StageType.max_with_const:                   MaxConst(),
        StageType.myriadX_convolution:              MyriadXHardwareLayer(),
        StageType.myriadX_fully_connected_layer:    MyriadXHardwareLayer(),
        StageType.myriadX_pooling:                  MyriadXHardwareLayer(),
        StageType.none:                             NoOp(),
        StageType.power:                            Power(),
        StageType.prelu:                            Prelu(),
        StageType.relu:                             Relu(),
        StageType.relu_x:                           Relu(),
        StageType.leaky_relu:                       Relu(),
        StageType.reshape:                          Reshape(),
        StageType.rsqrt:                            Rsqrt(),
        StageType.scale:                            Scale(),
        StageType.scale_with_scalar:                ScaleScalar(),
        StageType.sigmoid:                          Sigmoid(),
        StageType.soft_max:                         Softmax(),
        StageType.square:                           Square(),
        StageType.sum_reduce:                       SumReduce(),
        StageType.tanh:                             TanH(),
        StageType.toplanemajor:                     ToPlaneMajor(),
        StageType.permute:                          Permute(),
        StageType.permute_flatten:                  PermuteFlatten(),
        StageType.normalize:                        Normalize(),
        StageType.prior_box:                        PriorBox(),
        StageType.detection_output:                 DetectionOutput(),
        StageType.upsampling:                       Upsampling(),
    }

    if force_op:
        # In the new parser, we use in-place Operations, rather than postOperations.
        op_mapping[StageType.relu] = ReluOp()
        op_mapping[StageType.relu_x] = ReluOp()
        op_mapping[StageType.leaky_relu] = ReluOp()
        op_mapping[StageType.elu] = EluOp()

    val = op_mapping.get(op_type, -1)



    # print(op_type)

    if val == -1:
        throw_error(ErrorTable.StageTypeNotSupported, op_type)

    if val == -1 and op_type != StageType.none:
        print("WARNING TYPE NOT PRESENT IN DEFINITION LIBRARY: ", op_type)
    else:
        return val
