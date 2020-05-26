import numpy as np
import math
import cv2
import os
import sys

SSD_LABEL_NAME = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
            8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
            15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

MOBILENET_SSD_LABEL_NAME = {0: "bg", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
            8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
            16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
            27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 
            37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
            44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 
            54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
            63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
            76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 
            85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush" }


def saveListToFile(filePath, mylist):
    with open(filePath, 'w') as file_handler:
        for item in mylist:
            file_handler.write("{}\n".format(item))


def parse_output(output, image_path, actual, display_image=False):
    
    img = cv2.imread(image_path)
    w, h, c = img.shape
    arr_labels = [] # saved to disk

    res = output.astype(np.float32)
    max_proposal_count = int(res.shape[0] / 7)
    
    output_shape = [1,1,max_proposal_count,7] # SSD {1,1,200,7}, MobilenetSSD {1,1,100,7}
    object_size = output_shape[3]

    label_name = SSD_LABEL_NAME
    if max_proposal_count == 100:   # ssd/mobilenet-ssd use different label values
        label_name = MOBILENET_SSD_LABEL_NAME

    # Each detection has image_id that denotes processed image
    print("%12s %9s [%5s %5s %5s %5s]" % ("label", "score", "xmin", "ymin", "xmax", "ymax"))
    for cur_proposal in range(max_proposal_count):
        image_id = res[cur_proposal * object_size + 0]
        label = res[cur_proposal * object_size + 1]
        confidence = res[cur_proposal * object_size + 2]

        if (image_id < 0) or (confidence == 0.0):
            continue

        xmin = res[cur_proposal * object_size + 3] * w
        ymin = res[cur_proposal * object_size + 4] * h
        xmax = res[cur_proposal * object_size + 5] * w
        ymax = res[cur_proposal * object_size + 6] * h

        if confidence > 0.4:
            # Drawing only objects with >50% probability
            print("%12s %9f [%5.0f %5.0f %5.0f %5.0f]" % (label_name[label], confidence, xmin, ymin, xmax, ymax))
            cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)

            arr_labels.append(label_name[label])

            label_text = label_name[label] + " " + str("{0:.2f}".format(confidence))
            label_background_color = (70, 120, 70) # greyish green background for text
            label_text_color = (255, 255, 255)   # white text

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = int(xmin)
            label_top = int(ymin) - label_size[1]
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]

            cv2.rectangle(img, (label_left-1, label_top-5),(label_right+1, label_bottom+1), label_background_color, -1)
            cv2.putText(img, label_text, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    if (actual):
        saveListToFile(os.getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo/actual_inference_results.txt", arr_labels)
    else:
        saveListToFile(os.getenv("DLDT_HOME") + "/bin/intel64/Debug/inference_results.txt", arr_labels)

    if display_image:
        cv2.imshow('YOLO detection',img)
        while cv2.getWindowProperty('YOLO detection', cv2.WND_PROP_VISIBLE) > 0:
            keyCode = cv2.waitKey(100)
            if keyCode == 27: # ESC key
                break
        cv2.destroyAllWindows()


def main():
    output = np.fromfile(os.getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo/output-0.bin", dtype=np.float16)
    ref = np.fromfile(os.getenv("DLDT_HOME") + "/bin/intel64/Debug/output_cpu.bin", dtype=np.float32)
    
    display_image=False
    if len(sys.argv) > 2: # pass arg to display image
        display_image=sys.argv[2]

    if ref is not None:
        print("Expected Output:")
        parse_output(ref.astype(np.float32), sys.argv[1], False, display_image)
    
    if output is not None:
        print("Actual Output:")
        parse_output(output.astype(np.float32), sys.argv[1], True, display_image)

if __name__=='__main__':
    main()
