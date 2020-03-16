import sys
import numpy as np
import os

if len(sys.argv) != 2:
    print("please specify input file")
    exit(1)

res_u8 = np.fromfile(sys.argv[1], np.uint8)
res_fp16 = np.fromfile(sys.argv[1], np.float16)
res_fp32 = res_fp16.astype(float)

res = res_fp16
# print(res)
# print(np.argmax(res))
# print(res.argsort()[-10:][::-1])

top_k = np.argsort(res)[-10:][::-1]
np.savetxt(os.getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo/actual_inference_results.txt", top_k, fmt='%i')

exit(0)
