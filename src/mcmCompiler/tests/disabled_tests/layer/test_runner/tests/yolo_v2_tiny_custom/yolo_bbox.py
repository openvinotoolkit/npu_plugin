import numpy as np
import math
import cv2

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

def get_detection(boxes):
    w, h, c = (416,416,3)

    label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
        8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
        15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

    detection = []
    for box in boxes:
        xmin = max(0, (box[0] - box[2] / 2.0) * w)
        xmax = min(w, (box[0] + box[2] / 2.0) * w)
        ymin = max(0, (box[1] - box[3] / 2.0) * h)
        ymax = min(h, (box[1] + box[3] / 2.0) * h)
        label = label_name[box[4]]
        score = box[5]*box[6]
        detection.append([label, score, xmin, xmax, ymin, ymax])

    return detection

def parse_output(output):
    thresh = 0.6
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

    boxes = apply_nms(boxes, nms)

    return get_detection(boxes)

def parse_and_print(output_data):
    detection = parse_output(output_data)

    print("%12s %9s [%5s %5s %5s %5s]" % ("label", "score", "xmin", "xmax", "ymin", "ymax"))
    for obj in detection[:10]:
        print("%12s %9f [%5.0f %5.0f %5.0f %5.0f]" % (obj[0], obj[1], obj[2], obj[3], obj[4], obj[5]))

def parse_and_show(output_data, image_path):
    detection = parse_output(output_data)

    img = cv2.imread(image_path)

    for obj in detection:
        xmin = obj[2]
        xmax = obj[3]
        ymin = obj[4]
        ymax = obj[5]

        cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)

        label_text = obj[0] + " " + str("{0:.2f}".format(obj[1]))
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(xmin)
        label_top = int(ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]

        cv2.rectangle(img, (label_left-1, label_top-5),(label_right+1, label_bottom+1), label_background_color, -1)
        cv2.putText(img, label_text, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('YOLO detection',img)
    while cv2.getWindowProperty('YOLO detection', cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(100)
        if keyCode == 27: # ESC key
            break
    cv2.destroyAllWindows()
