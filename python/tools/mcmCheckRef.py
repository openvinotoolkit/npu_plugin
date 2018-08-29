import os, sys
import numpy as np
import argparse

base = os.environ.get('MDK_HOME')
assert base is not None, "Please set MDK_HOME environment variable"
sys.path.append(base+"projects/Fathom/src2/")

import Controllers.Globals as GLOBALS
from Controllers.Parsers.Caffe import CaffeParser
from Views.Validate import *
from Models.EnumDeclarations import ValidationStatistic

from blob_print import blob_format
from generate_image import gen_data

# parse command line arguments
parser = argparse.ArgumentParser(description="mcmCheckRef.py compares results from 2 npy files.\n")
parser.add_argument('--metric', type=str, const="ALL:256", default="ALL:1", help='set of accuracy metrics to use for error analysis', nargs='?')
parser.add_argument('--reference', dest='reference', type=str, help='reference results eg. ./expected.npy')
parser.add_argument('--result', dest='result', type=str, help='results under test e.g. ./resnet50_.npy')
args = parser.parse_args()

# CPP Blob run result file
blob_res = args.result
ref_res = args.reference
blob_np = np.load( blob_res )
expected_np = np.load( ref_res )

quit_code = validation(expected_np, blob_np, "NA", ValidationStatistic.accuracy_metrics, "NA", "NA")
quit(quit_code)
