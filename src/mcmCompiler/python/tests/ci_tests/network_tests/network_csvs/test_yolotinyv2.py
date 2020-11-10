import numpy as np
import os
import sys
import subprocess

def main():
    nroot = os.getenv("NZOO_ROOT", "../../../../../../migNetworkZoo")
    resnet = os.path.join(nroot, "tools/kmb_test_generation/yolo_tiny_v2.py")
    onnx_path = os.path.join(nroot, "public/yolo/ONNX/tiny_yolo_v2.onnx")

    cmd = "python3 " + resnet + " -c 3 -y 224 -x 224 --quantize --onnx_model "+ onnx_path
    cmd = cmd.split()
    code = subprocess.run(cmd)

    outfile = os.path.join(nroot, "internal/unit_tests/CompilerTestsKmb/layers/yolo_tiny_v2_weights/quantized_model.tflite")

    print("\nFILE:"+outfile, end="")

if __name__=='__main__':
    main()
