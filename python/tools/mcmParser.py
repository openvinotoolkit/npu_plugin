#! /usr/bin/env python3
import os
import sys
import argparse
import numpy as np

if sys.version_info[0] != 3:
    sys.stdout.write("Attempting to run with a version of Python != 3.x\n")
    sys.exit(1)

from parserTools.TensorFlowLiteParser import TensorFlowLite
from parserTools.CPPWrapper import ComposeForCpp

class Arguments:
    pass

def define_and_parse_args():

    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    parser.add_argument('--network-description', metavar='', type=str, nargs='?',
                        help='[0] Fathom Operational Mode')

    parser.add_argument('--produceMcmDescriptor', action="store_true", help="Produce the network in mcm form")

    parser.add_argument('--image', metavar='', type=str, nargs='?',
                    help='Image to use in operation')

    parser.add_argument('--comp-descriptor', type=str, action='store',
                    help="Path to user defined compilation descriptor file of MCM Compiler")

    args = parser.parse_args(namespace=Arguments)
    return args

if __name__ == "__main__":

    args = define_and_parse_args()
    p = TensorFlowLite.TensorFlowLiteParser()
    p.loadNetworkObjects(args.network_description)
    parsedLayers = p.parse()
    input_data, expected_result, output_tensor_name = p.calculateReference(args)
    ComposeForCpp(parsedLayers, args)
    quit()