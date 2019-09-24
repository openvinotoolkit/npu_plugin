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

from enum import Enum

class MemoryIndex(Enum):
    """
    These indexes define in which memory the pointers refer
    """
    none = 0
    input = 1
    output = 2
    blob = 3
    workbuffer = 4

class StageType(Enum):
    """
    Certain network stages may not be supported but are included for completeness.
    """
    convolution = 0
    max_pooling = 1
    average_pooling = 2
    soft_max = 3
    fully_connected_layer = 4
    none = 5
    relu = 6
    relu_x = 7
    depthwise_convolution = 8
    bias = 9
    prelu = 10
    LRN = 11
    eltwise_sum = 12
    eltwise_prod = 13
    eltwise_max = 14
    scale = 15
    relayout = 16
    square = 17
    innerlrn = 18
    copy = 19
    sigmoid = 20
    tanh = 21
    deconvolution = 22
    elu = 23
    reshape = 24
    toplanemajor = 25
    power = 26
    crop = 27
    tile = 28
    region = 29
    reorg = 30
    convertu8fp16 = 31
    convertf32f16 = 32
    myriadX_convolution = 33
    myriadX_pooling = 34
    myriadX_fully_connected_layer = 35
    myriadX_post_ops = 36
    storage_order_convert = 37
    permute = 38
    normalize = 39
    prior_box = 40
    detection_output = 41
    leaky_relu = 42
    sum_reduce = 43
    max_with_const = 44
    rsqrt = 45
    scale_with_scalar = 46
    permute_flatten = 48
    upsampling = 49

    # The following enums have not been officially assigned values
    dropout = 993
    maxout = 994
    normalizaton = 995
    r_relu = 996
    BNLL = 997
    abs = 998
    stochastic_pooling = 999




class TapsOrder(Enum):
    """
     H = kernel height, W = kernel width,
     C = input planes, K = output planes
    """
    orderHWCK = 0   # Used by MvTensor and Tensorflow
    orderKCHW = 1   # Used by Caffe and Torch


class PadStyle(Enum):
    """
    Certain pad styles may not be supported in all operations
    """
    none = 0
    tfvalid = 1
    caffe = 2
    tfsame = 3


class DataType(Enum):
    """
    Data types to be used in mvTensor. Not all supported,
    """
    fp16 = 0
    fp8 = 1
    int32 = 2

    # The following Enums have not been official assigned values.
    fp64 = 3
    fp32 = 4
    int64 = 5
    int16 = 6
    int8 = 7
    int4 = 8
    int2 = 9
    bit = 10
    chr = 11


class ErrorTable(Enum):
    Unknown = 0
    CaffeImportError = 1
    PythonVersionError = 2
    CaffeSyntaxError = 3
    StageTypeNotSupported = 4
    StageDetailsNotSupported = 5
    MyriadExeNotPresent = 6
    USBError = 7
    ArgumentErrorDescription = 8
    ArgumentErrorWeights = 9
    ModeSelectionError = 10
    ArgumentErrorExpID = 11
    ArgumentErrorImage = 12
    NoOutputNode = 13
    DataTypeNotSupported = 14
    ParserNotSupported = 15
    InputNotFirstLayer = 16
    GraphConstructionFailure = 17
    ConversionNotSupported = 18
    ArgumentErrorRequired = 19
    InputSyntaxNotSupported = 20
    ValidationSelectionError = 21
    UnrecognizedFileType = 22
    InvalidInputFile = 23
    AttemptedBatchMode = 24
    MyriadRuntimeIssue = 25
    NoUSBBinary = 26
    InvalidNumberOfShaves = 27
    CaffeMemoryError = 28
    TupleSyntaxWrong = 29
    InputFileUnsupported = 30
    USBDataTransferError = 31
    OptimizationParseError = 32
    NoTemperatureRecorded = 33
    TFNotEvaluated = 34
    NoResources = 35
    OutputNodeNameTopMismatch = 37
    InvalidNpyFile = 38
    InvalidTuple = 39
    InvalidMean = 40
    HardwareConfigurationError = 41
    no_check_optimization = 42
    InvalidLayout = 43
    MultipleInputs = 44
    TargetFileError = 45
    ResourceNotAvailable = 46
    ArgumentError = 47
    GitHooksWarning = 48


class Parser(Enum):
    """
    Potential parsers for the toolkit. Some of these will not be supported, either short-term or long-term
    """
    TensorFlow = 0
    Caffe = 1
    Torch = 2
    Theano = 3
    Debug = 4
    TensorFlowLite = 5
    Keras = 6
