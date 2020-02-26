import numpy as np
import math
import cv2
import os
import sys

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2):  # x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(yolo_boxes, nms):
    boxes = sorted(yolo_boxes, key=lambda d: d[7])[::-1]
    p = dict()

    res = list()
    for i in range(len(boxes)):
        if i in p:
            continue

        res.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if j in p:
                continue
            iou = cal_iou(boxes[i], boxes[j])
            if iou >= nms:
                p[j] = 1

    return res

def saveListToFile(filePath, mylist):
    with open(filePath, 'w') as file_handler:
        for item in mylist:
            file_handler.write("{}\n".format(item))

def parse_output(output, image_path, actual, display_image=False):
    thresh = 0.4
    nms = 0.2
    res = output.astype(np.float32)
    res = np.reshape(res,(125,13,13))
    swap = np.zeros((13, 13, 5, 25))

    for h in range(13):
        for w in range(13):
            for c in range(125):
                i = int(c / 25)
                j = c % 25
                swap[h][w][i][j]=res[c][h][w]

    biases = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] # yolo-v2-tiny
    # biases = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071] # yolo-v2

    boxes = list()
    for h in range(13):
        for w in range(13):
            for n in range(5):
                box = list()
                cls = list()
                x = (w + swap[h][w][n][0]) / 13.0
                y = (h + swap[h][w][n][1]) / 13.0
                ww = (math.exp(swap[h][w][n][2]) * biases[2 * n]) / 13.0
                hh = (math.exp(swap[h][w][n][3]) * biases[2 * n + 1]) / 13.0
                obj_score = swap[h][w][n][4]
                for p in range(20):
                    cls.append(swap[h][w][n][5 + p])

                box.append(x)
                box.append(y)
                box.append(ww)
                box.append(hh)
                box.append(cls.index(max(cls)) + 1)
                box.append(obj_score)
                box.append(max(cls))
                box.append(obj_score * max(cls))
                if box[5] * box[6] > thresh:
                    boxes.append(box)

    res = apply_nms(boxes, nms)
    label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
                  8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
                  15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

    img = cv2.imread(image_path)
    w, h, c = img.shape
    arr_labels = []
    print("%12s %9s [%5s %5s %5s %5s]" % ("label", "score", "xmin", "ymin", "xmax", "ymax"))
    for box in res[:10]: # top 10
        xmin = (box[0] - box[2] / 2.0) * w
        xmax = (box[0] + box[2] / 2.0) * w
        ymin = (box[1] - box[3] / 2.0) * h
        ymax = (box[1] + box[3] / 2.0) * h
        if xmin < 0:
            xmin = 0
        if xmax > w:
            xmax = w
        if ymin < 0:
            ymin = 0
        if ymax > h:
            ymax = h

        print("%12s %9f [%5.0f %5.0f %5.0f %5.0f]" % (label_name[box[4]], box[5]*box[6], xmin, ymin, xmax, ymax))
        cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)

        arr_labels.append(label_name[box[4]])

        label_text = label_name[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
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
    output = np.fromfile(os.getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo/output-0.bin", dtype=np.float16, count=13*13*125)
    ref = np.fromfile(os.getenv("DLDT_HOME") + "/bin/intel64/Debug/output_cpu.bin", dtype=np.float32, count=13*13*125)
    
    if ref is not None:
        print("Expected Output:")
        parse_output(ref.astype(np.float32), sys.argv[1], actual=False)
    
    if output is not None:
        print("Actual Output:")
        parse_output(output.astype(np.float32), sys.argv[1], actual=True) # len(sys.argv) > 2) # pass arg to display image

if __name__=='__main__':
    main()
