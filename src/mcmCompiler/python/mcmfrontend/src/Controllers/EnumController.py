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
import numpy as np
import warnings
from Models.EnumDeclarations import ErrorTable
from Models.EnumDeclarations import *


def stage_as_label(stage):
    d = {
        StageType.convolution: "conv",
        StageType.max_pooling: "maxpool",
        StageType.average_pooling: "avpool",
        StageType.deconvolution: "deconv",
        StageType.reshape: "reshape",
        StageType.power: "power",
        StageType.permute: "permute",
        StageType.normalize: "normalize",
        StageType.prior_box: "prior_box",
        StageType.detection_output: "detection_output",
    }
    if stage in d:
        return d[stage]
    return "no_optimization"

def completion_msg(mode):
    if mode == "generate":
        msg = "...Blob File Generated [OK]"
    elif mode == "profile":
        msg = "...PDF Report Created [OK]"
    else:
        msg = "...Complete: [OK]"

    return msg


def get_class_of_op(op):
    """
    Get type of operation
    :param op:
    :return:
    """

    class_dict = {
        StageType.convolution: "Convolution",
        StageType.depthwise_convolution: "Convolution",
        StageType.fully_connected_layer: "FCL",
        StageType.deconvolution: "Deconvolution",
        StageType.reshape: "Reshape",
        StageType.permute: "Permute",
        StageType.normalize: "Normalize",
        StageType.prior_box: "PriorBox",
        StageType.detection_output: "DetectionOutput"
    }
    if op in class_dict:
        return class_dict[op]
    else:
        return "Unknown"


def dtype_size(e):
    """
    Size in bytes. Used for buffer calculations
    :param e:
    :return:
    """

    if e == DataType.fp64:
        return 8
    if e == DataType.fp32:
        return 4
    if e == DataType.fp16 or e == 2:
        return 2
    if e == DataType.fp8:
        return 1
    if e == DataType.int64:
        return 8
    if e == DataType.int32:
        return 4
    if e == DataType.int16:
        return 2
    if e == DataType.int8:
        return 1
    if e == DataType.chr:
        return 1
    if e == DataType.int4:
        return 0.5
    if e == DataType.int2:
        return 0.25
    if e == DataType.bit:
        return 0.125


def dtype_as_enum(d):
    if d == np.float16:
        return DataType.fp16
    if d == np.float8:
        return DataType.fp8
    if d == np.int32:
        return DataType.int32


def enum_as_dtype(e):
    """
    Return a numpy type instead of our enums
    :param e:
    :return:
    """
    if e == DataType.fp64:
        return np.floatfp64
    if e == DataType.fp32:
        return np.float32
    if e == DataType.fp16 or e == 2:
        return np.float16
    if e == DataType.fp8:
        return np.float8
    if e == DataType.int64:
        return np.int64
    if e == DataType.int32:
        return np.int32
    if e == DataType.int16:
        return np.int16
    if e == DataType.int8:
        return np.int8
    if e == DataType.int4:
        return np.int4
    if e == DataType.int2:
        return np.int2
    if e == DataType.bit:
        return np.bit


# Warnings & Errors:

def setup_warnings():
    """
    Disables Line numbers being triggered
    :return:
    """
    # Force warnings.warn() to omit the source code line in the message
    formatwarning_orig = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
        formatwarning_orig(message, category, filename, lineno, line='')


def throw_warning(e, extra=None):
    msg = "[Warning: " + str(e.value) + "] "
    if e == ErrorTable.OptimizationParseError:
        msg += " Problem parsing optimization file. Using Defaults"
    if e == ErrorTable.OutputNodeNameTopMismatch:
        (name, top) = extra
        msg += " Output layer\'s name (" + str(name) + \
            ") must match its top (" + str(top) + ")"
    if e == ErrorTable.GitHooksWarning:
        msg += "Cannot install git hooks. Please run mcmFrontend.py with sudo (required only for the first run)"
    if e == ErrorTable.no_check_optimization:
        msg += "Warning: Optimization file is being used. You must delete the file if undesired."

    print("\033[93m" + str(msg) + "\033[0m", file=sys.stderr)


def compiler_assert(condition, e, extra=None):
    """
    If condition is false, throws custom error
    """
    if not condition:
        throw_error(e, extra=extra)


def throw_error(e, extra=None):
    msg = "[Error " + str(e.value) + "] "
    if e == ErrorTable.CaffeImportError:
        msg += "Setup Error: Caffe Import Error."
    if e == ErrorTable.CaffeSyntaxError:
        msg += "Setup Error: Caffe Syntax Error: " + str(extra)
    if e == ErrorTable.PythonVersionError:
        msg += "Setup Error: Using a version of Python that is unsupported."
    if e == ErrorTable.ModeSelectionError:
        msg += "Toolkit Error: No such Mode '" + str(extra) + "'."
    if e == ErrorTable.ArgumentErrorDescription:
        msg += "Argument Error: Network description cannot be found."
    if e == ErrorTable.ArgumentErrorWeights:
        msg += "Argument Error: Network weight cannot be found."
    if e == ErrorTable.ArgumentErrorImage:
        msg += "Argument Error: Image cannot be found."
    if e == ErrorTable.ArgumentErrorExpID:
        msg += "Argument Error: Expected ID not provided."
    if e == ErrorTable.NoUSBBinary:
        msg += "Toolkit Error: No moviUsbBoot Executable detected."
    if e == ErrorTable.USBError:
        msg += "Toolkit Error: USB Failure. Code: " + str(extra)
    if e == ErrorTable.MyriadExeNotPresent:
        msg += "Setup Error: no Myriad Executable detected."
    if e == ErrorTable.NoOutputNode:
        msg += "Toolkit Error: Provided OutputNode/InputNode name does not exist or does not match with one contained in " + \
               "model file Provided: " + str(extra)
    if e == ErrorTable.StageTypeNotSupported:
        msg += "Toolkit Error: Stage Type Not Supported: " + str(extra)
    if e == ErrorTable.StageDetailsNotSupported:
        msg += "Toolkit Error: Stage Details Not Supported: " + str(extra)
    if e == ErrorTable.DataTypeNotSupported:
        msg += "Toolkit Error: Data Type Not Supported: " + str(extra)
    if e == ErrorTable.ParserNotSupported:
        msg += "Toolkit Error: Parser Not Supported: " + str(extra)
    if e == ErrorTable.InputNotFirstLayer:
        msg += "Toolkit Error: Internal Error: Input Stage is not first layer."
    if e == ErrorTable.GraphConstructionFailure:
        msg += "Toolkit Error: Internal Error: Could not build graph. Missing link: " + \
            str(extra)
    if e == ErrorTable.ConversionNotSupported:
        msg += "Toolkit Error: Internal Error: Invalid Conversion Optimization. From: " + \
            str(extra)
    if e == ErrorTable.ArgumentErrorRequired:
        msg += "Toolkit Error: Setup Error: Not all required arguments were passed / Erroneous arguments."
    if e == ErrorTable.InputSyntaxNotSupported:
        msg += "Toolkit Error: Input Layer must be in an input_shape construct."
    if e == ErrorTable.ValidationSelectionError:
        msg += "Argument Error: Validation metric not supported " + str(extra)
    if e == ErrorTable.UnrecognizedFileType:
        msg += "Toolkit Error: Unable to tell what parser is required. Consider overriding with --parser argument."
    if e == ErrorTable.InvalidInputFile:
        msg += "Toolkit Error: Mismatch between input layer of network and given input file."
    if e == ErrorTable.AttemptedBatchMode:
        msg += "Toolkit Error: Inference with Batching of images is not supported."
    if e == ErrorTable.MyriadRuntimeIssue:
        msg += 'Myriad Error: "' + str(extra) + '".'
    if e == ErrorTable.InvalidNumberOfShaves:
        msg += "Setup Error: Too Many / Too Few Shave Processors Selected."
    if e == ErrorTable.CaffeMemoryError:
        msg += "Caffe Error: MemoryError. Potential Cause: Available RAM not sufficient for Network to be loaded into Caffe"
    if e == ErrorTable.TupleSyntaxWrong:
        msg += "Setup Error: Tuple Syntax Incorrect, should be in form x,y,z "
    if e == ErrorTable.InputFileUnsupported:
        msg += "Toolkit Error: Filetype not supported as a input."
    if e == ErrorTable.USBDataTransferError:
        msg += "USB Error: Problem Transferring data."
    if e == ErrorTable.OptimizationParseError:
        msg += "Setup Error: Problem parsing configuration File."
    if e == ErrorTable.NoTemperatureRecorded:
        msg += "Toolkit Error: No Temperature Read from device."
    if e == ErrorTable.TFNotEvaluated:
        msg += "Setup Error: Values for input contain placeholder. Pass an absolute value."
    if e == ErrorTable.NoResources:
        msg += "Setup Error: Not enough resources on Myriad to process this network."
    if e == ErrorTable.InvalidNpyFile:
        msg += "Toolkit Error: Unable to load .npy file (" + str(extra) + ")"
    if e == ErrorTable.InvalidTuple:
        msg += "Toolkit Error: Invalid tuple format (" + str(extra) + ")"
    if e == ErrorTable.InvalidMean:
        msg += "Toolkit Error: Invalid mean value (" + str(extra) + ")"
    if e == ErrorTable.HardwareConfigurationError:
        msg += "Toolkit Error: Invalid configuration of Hardware. (" + str(
            extra) + ")"
    if e == ErrorTable.InvalidLayout:
        msg += "Toolkit Error: Invalid layout (" + str(extra) + ")"
    if e == ErrorTable.MultipleInputs:
        msg += "Toolkit Error: Unsupported: Multiple graph inputs or Malformed graph connections."
    if e == ErrorTable.TargetFileError:
        msg += "Target File Error: " + str(extra)
    if e == ErrorTable.ResourceNotAvailable:
        msg += "Resource Not Available Error: " + str(extra)
    if e == ErrorTable.ArgumentError:
        msg += "Argument Error: " + str(extra)

    print("\033[91m" + str(msg) + "\033[0m", file=sys.stderr)
    sys.exit(1)
