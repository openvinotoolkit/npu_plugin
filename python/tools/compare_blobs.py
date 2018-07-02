import os, sys
import numpy as np
import argparse

base = os.environ.get('MDKPath')
sys.path.append(base+"projects/Fathom/src2/")

from Views.Validate import *
from Models.EnumDeclarations import ValidationStatistic


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

os.system('python3 ./run_blob.py ' + blob1_path + ' \(1,224,224,3\) \(112,112,64\) -i test.jpg -res ' + blob1_res)
os.system('python3 ./run_blob.py ' + blob2_path + ' \(1,224,224,3\) \(112,112,64\) -i test.jpg -res ' + blob2_res)

blob1_np = np.load('./' + blob1_res + '.npy')
blob2_np = np.load('./' + blob2_res + '.npy')

validation(blob1_np, blob2_np, "NA", ValidationStatistic.accuracy_metrics, "NA", "NA")