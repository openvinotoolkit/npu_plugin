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

import os
import argparse
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable
from Models.EnumDeclarations import Parser
import Controllers.Globals as GLOBALS
import Models.Layouts as Layouts


class mcmFrontendArguments:
    """
    Container for mcmFrontend Arguments.
    Required for using mcmFrontend as a module (dev scripts)
    """

    def copy_fields(self, argObj):
        for var in vars(argObj):
            att_val = getattr(argObj, var)
            setattr(self, var, att_val)

def usage_msg(name=None):
    return '''mcmFrontend
         Show this help:
            mcmFrontend help
         Version:
            mcmFrontend version

         mcmFrontend ...

            * --network-description [...]
            * --comp-descriptor [...]
        '''


def define_and_parse_args():
    # Argument Checking
    parser = argparse.ArgumentParser(
        description="""mcmFrontend is Movidius\'s machine learning software framework. \n
mcmFrontend converts trained offline neural networks into embedded neural networks running on the \
ultra-low power Kmb VPU.\nBy targeting Kmb, \
mcmFrontend makes it easy to profile, tune and optimize your standard TensorFlow, TensorFlowLite or Caffe neural network. """,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage_msg())

    # Source of Network Configurations
    parser.add_argument(
        '--network-description',
        metavar='',
        type=str,
        nargs='?',
        help='Relative path to network description file. Typical usage is for caffe prototxt file' +
        ' or TensorFlow .pb file')
    parser.add_argument(
        '--comp-descriptor',
        type=str,
        action='store',
        help="Path to user defined compilation descriptor file of MCM Compiler")
    try:
        args = parser.parse_args()
    except BaseException:
        throw_error(ErrorTable.ArgumentErrorRequired)

    GLOBALS.USING_KMB =  True
    args.net_description = path_arg(args.network_description)
    if args.net_description is not None:
        args.parser = Parser.TensorFlowLite
    fa = mcmFrontendArguments()
    fa.copy_fields(args)

    return fa


def path_arg(path):
    if path is not None:
        return os.path.normpath(path)
    else:
        return None

def path_check(path, error):
    if path is not None and os.path.isfile(path):
        return True
    else:
        throw_error(error)
