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

#pragma once
#include <string>
#include <vector>

namespace postprocess{
    int yolov2(const float *data, std::size_t * shape4D, std::size_t * strides4D, float thresh, float nms,
            std::size_t num_classes, int image_width, int image_height, int net_width, int net_height,
            float * result);

    const std::vector<std::string> YOLOV2_TINY_LABELS = {
         "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
         "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
         "sofa", "train", "tvmonitor"
    };
};
