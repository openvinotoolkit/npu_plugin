//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ie_core.hpp"
#include "model_helper.h"

class ModelSqueezenetV1_1_Helper : public ModelHelper {
public:
    explicit ModelSqueezenetV1_1_Helper();
};

//------------------------------------------------------------------------------
inline ModelSqueezenetV1_1_Helper::ModelSqueezenetV1_1_Helper() {
    _modelRelatedPath =
            "KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8";
    loadModel();
}
