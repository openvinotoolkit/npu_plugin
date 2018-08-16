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
args = parser.parse_args()

# NCCheck Files:
blob1_path = args.blob1
blob1_res = 'Blob1_result'

# CPP Blob run result file
blob2_path = args.blob2
blob2_res = 'Blob2_result'

# Get blob inputs and outputs.
a = blob_format.parse_file(blob1_path)
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


os.system('python3 ./run_blob.py ' + blob1_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\) -i test.npy -res ' + blob1_res)
os.system('python3 ./run_blob.py ' + blob2_path + ' \(1,' +in_y+','+in_x+','+in_z+'\) \('+out_y+','+out_x+','+out_z+'\)  -i test.npy -res ' + blob2_res)

blob1_np = np.load('./' + blob1_res + '.npy')
blob2_np = np.load('./' + blob2_res + '.npy')

validation(blob1_np, blob2_np, "NA", ValidationStatistic.accuracy_metrics, "NA", "NA")
