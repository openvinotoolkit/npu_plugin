import numpy as np
import os
import sys
import subprocess

def main():
    nroot = os.getenv("NZOO_ROOT", "../../../../../../migNetworkZoo")

    outfile = os.path.join(nroot, "public/MobileNet/TfLite/mobilenet_v2_1.0_224_quant_no_reshape.tflite")

    print("\nFILE:"+outfile, end="")

if __name__=='__main__':
    main()
