//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <stdlib.h>
#include <math.h>
#include "region_yolov2tiny.h"

namespace postprocess {

#define MAX_CLASSES 128
const float YOLOV2_TINY_ANCHORS[10] = {1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f};
// const float YOLOV2_ANCHORS_80_CLASSES[10] = {0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f};

// const char * YOLOV2_TINY_LABELS[20] = {
//         "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
//         "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
//         "sofa", "train", "tvmonitor"
//     };

typedef struct {
  // the (x,y) is the center of the bounding box
  // rather than top-left corner
  float x, y, w, h;
  float probs[MAX_CLASSES];
} box;

typedef struct {
  float x0, y0, x1, y1;
  float conf;
  int label;
} detect_obj;

typedef struct {
  int index;
  int cclass;
  float *probs;
} sortable_bbox;

static void get_region_boxes(const float *predictions, int *shape4D,
                             int *strides4D, int num_classes, float thresh,
                             box *boxes, const float *anchors);
static float box_iou(box a, box b);

static int nms_comparator(const void *pa, const void *pb) {
  sortable_bbox a = *reinterpret_cast<const sortable_bbox *>(pa);
  sortable_bbox b = *reinterpret_cast<const sortable_bbox *>(pb);
  float diff = a.probs[b.cclass] - b.probs[b.cclass];
  if (diff < 0.0f)
    return 1;
  else if (diff > 0.0f)
    return -1;
  return 0;
}

void do_nms_sort(box *boxes, int total, int classes, float thresh) {
  int i, j, k;
  sortable_bbox *s =
      reinterpret_cast<sortable_bbox *>(calloc(total, sizeof(sortable_bbox)));

  for (i = 0; i < total; ++i) {
    s[i].index = i;
    s[i].cclass = 0;
    s[i].probs = boxes[i].probs;
  }

  for (k = 0; k < classes; ++k) {
    for (i = 0; i < total; ++i) {
      s[i].cclass = k;
    }
    qsort(s, total, sizeof(sortable_bbox), nms_comparator);
    for (i = 0; i < total; ++i) {
      if (s[i].probs[k] == 0)
        continue;
      box a = boxes[s[i].index];
      for (j = i + 1; j < total; ++j) {
        box b = boxes[s[j].index];
        if (box_iou(a, b) > thresh) {
          s[j].probs[k] = 0;
        }
      }
    }
  }
  free(s);
}

//========================= BOX IOU ===================================
static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0)
    return 0;
  float area = w * h;
  return area;
}

static float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

static float box_iou(box a, box b) {
  return box_intersection(a, b) / box_union(a, b);
}

int max_index(float *a, int n) {
  if (n <= 0)
    return -1;
  int i, max_i = 0;
  float max = a[0];
  for (i = 1; i < n; ++i) {
    if (a[i] > max) {
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth,
                          int relative) {
  int i;
  int new_w = 0;
  int new_h = 0;

  if ((static_cast<float>(netw) / w) < (static_cast<float>(neth) / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  for (i = 0; i < n; ++i) {
    box b = boxes[i];
    b.x = (b.x - (netw - new_w) / 2. / netw) / (static_cast<float>(netw) / netw);
    b.y = (b.y - (neth - new_h) / 2. / neth) / (static_cast<float>(new_h) / neth);
    b.w *= static_cast<float>(netw) / new_w;
    b.h *= static_cast<float>(neth) / new_h;
    if (!relative) {
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
    }
    boxes[i] = b;
  }
}

static float _sigmoid(float x) { return 1. / (1. + exp(-x)); }

static void _softmax(float *x, int cnt) {
  const float t0 = -100.0f;
  float max = x[0], min = x[0];
  for (int i = 1; i < cnt; i++) {
    if (min > x[i])
      min = x[i];
    if (max < x[i])
      max = x[i];
  }

  for (int i = 0; i < cnt; i++) {
    x[i] -= max;
  }

  if (min < t0) {
    for (int i = 0; i < cnt; i++)
      x[i] = x[i] / min * t0;
  }

  // normalize as probabilities
  float expsum = 0;
  for (int i = 0; i < cnt; i++) {
    x[i] = exp(x[i]);
    expsum += x[i];
  }
  for (int i = 0; i < cnt; i++) {
    x[i] = x[i] / expsum;
  }
}

// predictions is a tensor of shape (13,13,5,25), but stride is 128,
static void get_region_boxes(const float *predictions, int *shape4D,
                             int *strides4D, int num_classes, float thresh,
                             box *boxes, const float *anchors) {
  int i, j, n;
  int lh = shape4D[0];
  int lw = shape4D[1];
  int num_anchor = shape4D[2];
  int num_entry = shape4D[3];
  float raw_netout[MAX_CLASSES + 5];
// LAYOUT of predictions
// by cell, by (loc_4,conf_1,probs_classes), by anchor
#define PRED_IDX(anchor_idx, cell_idx, entry_idx)                              \
    (cell_idx * strides4D[1] + anchor_idx * strides4D[2] +                     \
    entry_idx * strides4D[3])

#define BOX_IDX(anchor_idx, cell_idx) (anchor_idx * (lw * lh) + cell_idx)

  int kkk = 0;
  for (i = 0; i < lw * lh; ++i) {
    int row = i / lw;
    int col = i % lw;
    for (n = 0; n < num_anchor; ++n) {
      int index = BOX_IDX(n, i);
      box *pb = &boxes[index];
      // LAYOUT of boxes:
      // by cell then by anchor
      for (j = 0; j < num_entry; ++j)
        raw_netout[j] = predictions[PRED_IDX(n, i, j)];
      _softmax(raw_netout + 5, num_classes);
      raw_netout[4] = _sigmoid(raw_netout[4]);
      for (j = 0; j < num_classes; ++j) {
        raw_netout[5 + j] *= raw_netout[4];
        if (raw_netout[5 + j] <= thresh)
          raw_netout[5 + j] = 0;
      }
      kkk++;

      pb->x = (col + _sigmoid(raw_netout[0])) / lw;
      pb->y = (row + _sigmoid(raw_netout[1])) / lh;
      pb->w = exp(raw_netout[2]) * anchors[2 * n] / lw;
      pb->h = exp(raw_netout[3]) * anchors[2 * n + 1] / lh;

      float prob_sum = 0;
      for (j = 0; j < num_classes; ++j) {
        pb->probs[j] = raw_netout[5 + j];
        prob_sum += pb->probs[j];
      }

      // pb->probs[j] = max;
    }
  }
  // correct_region_boxes(boxes, lw * lh * lnum, w, h, netw, neth, relative);
}

int yolov2(const float *data, int *shape4D, int *strides4D,
           float thresh, float nms, int num_classes,
           int image_width, int image_height, float *result) {
  int lh = shape4D[0];
  int lw = shape4D[1];
  int num_anchor = shape4D[2];
  // int num_entry  = shape4D[3];
  const float *anchors = YOLOV2_TINY_ANCHORS;

  if (data == nullptr || anchors == nullptr) {
    return -1;
  }
  box *boxes = reinterpret_cast<box *>(malloc(lw * lh * num_anchor * sizeof(box)));

  get_region_boxes(data, shape4D, strides4D, num_classes, thresh, boxes,
                   anchors);
  do_nms_sort(boxes, lw * lh * num_anchor, num_classes, nms);

  int netw = 416;
  int neth = 416;
  float scalew = static_cast<float>(netw) / image_width;
  float scaleh = static_cast<float>(neth) / image_height;
  float scale = scalew < scaleh ? scalew : scaleh;
  float new_width = image_width * scale;
  float new_height = image_height * scale;
  float pad_w = (netw - new_width) / 2.0;
  float pad_h = (neth - new_height) / 2.0;
  // fill final result array
  int k = 0;
  for (int i = 0; i < lw * lh * num_anchor; ++i) {
    for (int j = 0; j < num_classes; j++) {
      box &b = boxes[i];
      if (b.probs[j] == 0)
        continue;

      float xmin = b.x - b.w / 2.0;
      float ymin = b.y - b.h / 2.0;
      float xmax = xmin + b.w;
      float ymax = ymin + b.h;

      xmin = (xmin * netw - pad_w) / scale;
      if (xmin < 0)
        xmin = 0;
      xmax = (xmax * netw - pad_w) / scale;
      if (xmax >= image_width)
        xmax = image_width - 1;
      ymin = (ymin * neth - pad_h) / scale;
      if (ymin < 0)
        ymin = 0;
      ymax = (ymax * neth - pad_h) / scale;
      if (ymax >= image_height)
        ymax = image_height - 1;

      // image_id, label, confidence, xmin, ymin, xmax, ymax
      result[k * 7 + 0] = 0;
      result[k * 7 + 1] = j;
      result[k * 7 + 2] = b.probs[j];
      result[k * 7 + 3] = xmin;
      result[k * 7 + 4] = ymin;
      result[k * 7 + 5] = xmax;
      result[k * 7 + 6] = ymax;

      k++;
    }
  }

  free(boxes);
  return k;
}
};  // namespace postprocess