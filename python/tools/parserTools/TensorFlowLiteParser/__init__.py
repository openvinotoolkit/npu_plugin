# Flatbuffer checking
import os

try:

    this_files_location = os.path.dirname(os.path.abspath(__file__))
    lib_path = this_files_location + "/../../../schema/graphFile/"

    if (not os.path.isfile(this_files_location + "/TensorFlowLiteParser/tflite/Model.py")) or \
        (not os.path.isfile(this_files_location + "/TensorFlowLiteParser/tflite/LeakyReluOptions.py")): # Detect an update of the schema...
        print("Compiling TFLite parser....")
        cmd = "flatc --python  -o "+this_files_location+"/ " +\
                                this_files_location+"/schema.fbs"
        print("Executing {}".format(cmd))
        ret = os.system(cmd)
        if ret != 0:
            print("Warning: Flatbuffers not installed. ")
            print("Please take the latest release from: " +
                "https://github.com/google/flatbuffers/releases and " +
                "ensure it is installed")
            sys.exit(1)

except Exception as e:
    print (e)
    print("Impossible to load TFLite parser code")
    quit()

import parserTools.TensorFlowLiteParser.Convolution
# import parserTools.TensorFlowLiteParser.Placeholder
import parserTools.TensorFlowLiteParser.NoOp
import parserTools.TensorFlowLiteParser.ReLU
import parserTools.TensorFlowLiteParser.Pooling
import parserTools.TensorFlowLiteParser.Eltwise
import parserTools.TensorFlowLiteParser.LRN
import parserTools.TensorFlowLiteParser.MatMul
# import parserTools.TensorFlowLiteParser.BiasAdd
import parserTools.TensorFlowLiteParser.Concat
# import parserTools.TensorFlowLiteParser.Reshape
# import parserTools.TensorFlowLiteParser.FusedBatchNorm
import parserTools.TensorFlowLiteParser.Softmax
import parserTools.TensorFlowLiteParser.Tanh
import parserTools.TensorFlowLiteParser.Sigmoid
# import parserTools.TensorFlowLiteParser.Identity
# import parserTools.TensorFlowLiteParser.Slice
import parserTools.TensorFlowLiteParser.Pad
# import parserTools.TensorFlowLiteParser.RealDiv

