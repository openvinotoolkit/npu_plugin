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

from Controllers.Parsers.ParserFactory import ParserFactory
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable
from Controllers.CPPWrapper import composeForCpp
import os

def load_network(arguments):
    """
    Loads the information contained in a network description file into our internal representation.
    It does not calculate buffer pointers for myriad. This is done in a seperate step.
    :param weights:
    :param path: absolute path to network description file
    :param parser: enum indicating which parser should be used.
    :return: Network with attributes populated, bar inputs and outputs.
    """

    parser = arguments.parser
    try:
        p = ParserFactory.getInstance(parser)
    except Exception as e:
        print(
            "Impossible to import {0} parser. Please verify that {0} is in your python PATH".format(
                parser.name))
        print("Error: {}".format(e))
        throw_error(ErrorTable.ParserNotSupported, parser.name)

    p.loadNetworkObjects(arguments.net_description)
    parsedLayers = p.parse(arguments)
    composeForCpp(parsedLayers, arguments)
    if os.path.exists("mcmNetwork.cpp"):
        os.remove("mcmNetwork.cpp")
    
