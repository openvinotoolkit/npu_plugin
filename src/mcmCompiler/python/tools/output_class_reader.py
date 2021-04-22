import sys
import numpy as np
import os

if len(sys.argv) != 4:
    print("please specify input files (actual, expected and the output precision)")
    exit(1)

# actual
res_u8 = np.fromfile(sys.argv[1], np.uint8)
res_fp16 = np.fromfile(sys.argv[1], np.float16)
res_fp32 = np.fromfile(sys.argv[1], np.float32)

if sys.argv[3].upper() == "FP32" : 
    res = res_fp32
    print("Using fp32 dtype...\n")
else: 
    res = res_fp16
    print("Using fp16 dtype...\n")

top_k = np.argsort(res)[-10:][::-1]
np.savetxt(os.getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo/actual_inference_results.txt", top_k, fmt='%i')

# expected
exp_fp32 = np.fromfile(sys.argv[2], np.float32)

exp_top_k = np.argsort(exp_fp32)[-10:][::-1]
np.savetxt(os.getenv("OPENVINO_HOME") + "/bin/intel64/" + os.getenv("BUILD_TYPE") + "/inference_results.txt", exp_top_k, fmt='%i')

exit(0)
