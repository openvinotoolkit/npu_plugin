import os, sys
import numpy as np
import argparse

base = os.environ.get('MDK_HOME')
assert base is not None, "Please set MDK_HOME environment variable"
sys.path.append(os.path.join(base, "projects/Fathom/src2/"))

import Controllers.Globals as GLOBALS
from Controllers.Parsers.Caffe import CaffeParser
from Views.Validate import *
from Models.EnumDeclarations import ValidationStatistic

from blob_print import blob_format
from generate_image import gen_data
from generate_image import gen_image

class netArgs:
    def __init__(self, network, image, inputnode, outputnode, inputsize, nshaves, weights, extargs):
        self.net_description = network
        filetype = network.split(".")[-1]
        self.parser = Parser.TensorFlow
        if filetype in ["prototxt"]:
            self.parser = Parser.Caffe
            if weights is None:
                weights = network[:-8] + 'caffemodel'
                if not os.path.isfile(weights):
                    weights = None
        self.conf_file = network[:-len(filetype)] + 'conf'
        if not os.path.isfile(self.conf_file):
            self.conf_file = None
        self.net_weights = weights
        self.input_node_name = inputnode
        self.output_node_name = outputnode
        self.input_size = inputsize
        self.number_of_shaves = nshaves
        self.image = image
        self.raw_scale = 1
        self.mean = None
        self.channel_swap = None
#        self.explicit_concat = extargs.explicit_concat
        self.acm = 0
        self.timer = None
        self.number_of_iterations = 2
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 1
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'validation'
        self.outputs_name = 'output'
        self.blob_name = 'cpp.blob'
        self.save_input = None
        self.save_output = None
        self.device_no = None
        self.exp_id = None
        self.enable_query = False
#        self.new_parser = extargs.new_parser
        self.cpp = False
        self.seed = -1
        self.accuracy_table = {}

# define command line arguments
parser = argparse.ArgumentParser(description="mcmGenRef.py produces a Fathom_expected.npy by performing inference on image supplied in the provided caffe model.\n")
parser.add_argument('--weights', dest='weights', type=str, help='Weights file, e.g. ./resnet.caffemodel')
parser.add_argument('--network', dest='network', type=str, help='Network model file, e.g. ./resnet.prototxt')
parser.add_argument('--image', dest='image', type=str, help='input image file, e.g. ./picture.png')
args = parser.parse_args()

# Load From Framework
calcArgs = netArgs(args.network, args.image, None, None, None, '1', args.weights, None)

#If there is no user defined image then generate a random one of the same size as the input tensor
if calcArgs.image is None:
    print("A user supplied image was not provided \n")
    print("Get input size of tensor\n")
    a = blob_format.parse_file(calcArgs.blob_name)
    
    in_x = a["Layers..."][1]["Op..."]["Buffers..."][0]["x"]
    in_y = a["Layers..."][1]["Op..."]["Buffers..."][0]["y"]
    in_z = a["Layers..."][1]["Op..."]["Buffers..."][0]["z"]

    print("Input size of tensor is:")
    print("In_y ", in_y)
    print("In_x ", in_x)
    print("In_z ", in_z)
    
    print("Generating a random numpy image file test.png\n")
    gen_image(in_y, in_x, in_z)
    
    #Set input image to be test.png    
    calcArgs.image = "test.png"
    print("calArgs clas varibale is ",calcArgs.image)
    print("Generating expected outcome with a randomly generated image test.png\n")
    p = CaffeParser()
    nw_file = args.network
    wt_file = args.weights
    p.loadNetworkObjects(nw_file, wt_file)
    input_data, expected_result, output_tensor_name = p.calculateReference(calcArgs)
    print("Wrote Fathom_expected.npy to file")
    
else:
    # Load From Framework
    print("Generating expected outcome with user defined image\n")
    calcArgs = netArgs(args.network, args.image, None, None, None, '1', args.weights, None)
    p = CaffeParser()
    nw_file = args.network
    wt_file = args.weights
    p.loadNetworkObjects(nw_file, wt_file)
    input_data, expected_result, output_tensor_name = p.calculateReference(calcArgs)
