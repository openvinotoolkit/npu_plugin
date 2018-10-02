#!/usr/bin/python3

import os, sys
import numpy as np
import argparse

base = os.environ.get('MDK_HOME')
assert base is not None, "Please set MDK_HOME environment variable"

sys.path.append(os.path.join(base, "projects/Fathom/src2/"))

from Views.Validate import *
from Models.EnumDeclarations import ValidationStatistic

from blob_print import blob_format
from generate_image import gen_data

parser = argparse.ArgumentParser(description="""Compare Two blob files.""",
                                    formatter_class=argparse.RawTextHelpFormatter, add_help=False)
parser.add_argument('blob1', metavar='', type=str, nargs='?',
                    help='path to first blob')
parser.add_argument('blob2', metavar='', type=str, nargs='?',
                    help='path to second blob')
parser.add_argument('--img', action='store_true',
                    help='Use a real image rather than rand data')
args = parser.parse_args()

# NCCheck Files:
blob1_path = args.blob1
blob1_res = 'Blob1_result'

# CPP Blob run result file
blob2_path = args.blob2
blob2_res = 'Blob2_result'

# Get blob inputs and outputs.
a = blob_format.parse_file(blob1_path)
Ain_x = a["Layers..."][1]["Op..."]["Buffers..."][0]["x"]
Ain_y = a["Layers..."][1]["Op..."]["Buffers..."][0]["y"]
Ain_z = a["Layers..."][1]["Op..."]["Buffers..."][0]["z"]

b = blob_format.parse_file(blob2_path)
Bin_x = b["Layers..."][1]["Op..."]["Buffers..."][0]["x"]
Bin_y = b["Layers..."][1]["Op..."]["Buffers..."][0]["y"]
Bin_z = b["Layers..."][1]["Op..."]["Buffers..."][0]["z"]

out_x = a["Layers..."][-1]["Op..."]["Buffers..."][1]["x"]
out_y = a["Layers..."][-1]["Op..."]["Buffers..."][1]["y"]
out_z = a["Layers..."][-1]["Op..."]["Buffers..."][1]["z"]

gen_data(Ain_y, Ain_x, Ain_z)
gen_data(Bin_y, Bin_x, Bin_z, post_str="B")

Ain_x = str(Ain_x)
Ain_y = str(Ain_y)
Ain_z = str(Ain_z)
Bin_x = str(Bin_x)
Bin_y = str(Bin_y)
Bin_z = str(Bin_z)

out_x = str(out_x)
out_y = str(out_y)
out_z = str(out_z)

if args.img:
    test_inputA = "test.npy"
else:
    test_inputA = "nps_mug.png"

if args.img:
    test_inputB = "testB.npy"
else:
    test_inputB = "nps_mugB.png"


test_inputB = test_inputA = "Debug"

os.system('python3 ./run_blob.py ' + blob1_path + ' \(1,' +Ain_y+','+Ain_x+','+Ain_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i '+test_inputA+' -res ' + blob1_res)
os.system('python3 ./run_blob.py ' + blob2_path + ' \(1,' +Bin_y+','+Bin_x+','+Bin_z+'\) \('+out_y+','+out_x+','+out_z+'\)  -i '+test_inputB+' -res ' + blob2_res)

blob1_np = np.load('./' + blob1_res + '.npy')
blob2_np = np.load('./' + blob2_res + '.npy')


validation(blob1_np, blob2_np, "NA", ValidationStatistic.accuracy_metrics, "NA", "NA")
