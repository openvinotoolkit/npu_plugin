//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

namespace Models {

struct ModelDesc {
    char pathToModel[FILENAME_MAX];
    size_t width;
    size_t height;
};

const ModelDesc squeezenet1_1 = {
        "/KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8",
        227,
        227
};

const ModelDesc googlenet_v1 = {
        "/KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8",
        224,
        224
};

const ModelDesc yolov3 = {
        "/KMB_models/INT8/public/yolo_v3/yolo_v3_tf_dense_int8_IRv10",
        416,
        416
};
}
