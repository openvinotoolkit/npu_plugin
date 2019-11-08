# Flatbuffer checking
import os
import sys

try:

    this_files_location = os.path.dirname(os.path.abspath(__file__))
    lib_path = this_files_location + "/../../lib/graphFile-schema/"

    if (not os.path.isfile(this_files_location + "/../TensorFlowLiteParser/tflite/Model.py")) \
        or (not os.path.isfile(this_files_location + "/../TensorFlowLiteParser/tflite/LeakyReluOptions.py")) \
        or (not os.path.isfile(this_files_location + "/../TensorFlowLiteParser/tflite/MeanOptions.py")):  # Detect an update of the schema...
        print("Compiling TFLite parser....")
        cmd = "flatc --python  -o " + this_files_location + "/ " +\
            this_files_location + "/schema.fbs"
        print("Executing {}".format(cmd))
        ret = os.system(cmd)
        if ret != 0:
            print("Warning: Flatbuffers not installed. ")
            print("Please take the latest release from: " +
                  "https://github.com/google/flatbuffers/releases and " +
                  "ensure it is installed")
            sys.exit(1)

except Exception as e:
    print(e)
    print("Impossible to load TFLite parser code")
    quit()

import Controllers.Parsers.TensorFlowLiteParser.Convolution
# import Controllers.Parsers.TensorFlowLiteParser.Placeholder
import Controllers.Parsers.TensorFlowLiteParser.NoOp
import Controllers.Parsers.TensorFlowLiteParser.ReLU
import Controllers.Parsers.TensorFlowLiteParser.Pooling
import Controllers.Parsers.TensorFlowLiteParser.Eltwise
import Controllers.Parsers.TensorFlowLiteParser.LRN
import Controllers.Parsers.TensorFlowLiteParser.MatMul
# import Controllers.Parsers.TensorFlowLiteParser.BiasAdd
import Controllers.Parsers.TensorFlowLiteParser.Concat
import Controllers.Parsers.TensorFlowLiteParser.Reshape
# import Controllers.Parsers.TensorFlowLiteParser.FusedBatchNorm
import Controllers.Parsers.TensorFlowLiteParser.Softmax
import Controllers.Parsers.TensorFlowLiteParser.Tanh
import Controllers.Parsers.TensorFlowLiteParser.Sigmoid
# import Controllers.Parsers.TensorFlowLiteParser.Identity
# import Controllers.Parsers.TensorFlowLiteParser.Slice
import Controllers.Parsers.TensorFlowLiteParser.Pad
# import Controllers.Parsers.TensorFlowLiteParser.RealDiv
import Controllers.Parsers.TensorFlowLiteParser.Mean