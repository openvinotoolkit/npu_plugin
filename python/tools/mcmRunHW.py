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

# parse command line arguments
parser = argparse.ArgumentParser(description="mcmCRunHW.py runs a blob on the HW, generating a results file\n")
parser.add_argument('--result', dest='result', type=str, help='results file. ./inference_result.npy')
parser.add_argument('--blob', dest='blob', type=str, help='blob file. ./renet50.blob')
parser.add_argument('--image', dest='image', type=str, help='input image file, e.g. ./picture.png')
args = parser.parse_args()

# CPP Blob run result file
blob_res = args.result
blob_path = args.blob
image_path = args.image

# Get blob inputs and outputs.
#
a = blob_format.parse_file(blob_path)

in_x = a["Layers..."][1]["Op..."]["Buffers..."][0]["x"]
in_y = a["Layers..."][1]["Op..."]["Buffers..."][0]["y"]
in_z = a["Layers..."][1]["Op..."]["Buffers..."][0]["z"]

out_x = a["Layers..."][-1]["Op..."]["Buffers..."][1]["x"]
out_y = a["Layers..."][-1]["Op..."]["Buffers..."][1]["y"]
out_z = a["Layers..."][-1]["Op..."]["Buffers..."][1]["z"]

gen_data(in_y, in_x, in_z)

in_x = str(in_x)
in_y = str(in_y)
in_z = str(in_z)

out_x = str(out_x)
out_y = str(out_y)
out_z = str(out_z)

GLOBALS.USING_MA2480 = True
os.system('python3 ./run_blob.py ' + blob_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i '+image_path+' -res '+blob_res)
