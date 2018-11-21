import os, sys
import numpy as np
import argparse
import subprocess
from _dummy_thread import exit

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

#Already generated test.npy
gen_data(in_y, in_x, in_z)

in_x = str(in_x)
in_y = str(in_y)
in_z = str(in_z)

print("In x :", in_x)
print("In y :", in_y)
print("In z :", in_z)

out_x = str(out_x)
out_y = str(out_y)
out_z = str(out_z)

print("out x :", out_x)
print("out y :", out_y)
print("out z :", out_z)

GLOBALS.USING_MA2480 = True
#os.system('python3 $MCM_HOME/python/tools/run_blob.py ' + blob_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i '+image_path+' -res '+blob_res)
#Using Subprocess() as it returns better error codes

#User supplied image not provided setting image_path to test.png
if image_path is None:
    print("User supplied image not provided setting image_path to test.png\n")
    image_path = "test.png"
    result = subprocess.call('python3 $MCM_HOME/python/tools/run_blob.py ' + blob_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i '+image_path+' -res '+blob_res, shell=True)

#Else a user supplied image was provided
else:
    result = subprocess.call('python3 $MCM_HOME/python/tools/run_blob.py ' + blob_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i '+image_path+' -res '+blob_res, shell=True)

if result != 0:
    sys.exit(-1) #return -1 on error running on hardware

sys.exit(0) #else return 0





