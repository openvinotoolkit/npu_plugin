import yolo_bbox
import numpy as np

import sys
import os.path

output = np.fromfile("output-0.bin", dtype=np.float16, count=13*13*125).astype(np.float32)
if (len(sys.argv) == 1):
    yolo_bbox.parse_and_print(output)
    yolo_bbox.parse_and_show(output, "./tests/yolo_v2_tiny_custom/person.bmp")
else:
    print("Run 'python " + sys.argv[0] + "' to show an image")
    reference_output = sys.argv[1]
    if not os.path.isfile(reference_output):
        print("Reference file '" + reference_output + "' does not exists")
        exit()

    ref = np.fromfile(reference_output, dtype=np.float16, count=13*13*125).astype(np.float32)

    print("Yolo output:")
    yolo_bbox.parse_and_print(output)
    print("Reference:")
    yolo_bbox.parse_and_print(ref)
