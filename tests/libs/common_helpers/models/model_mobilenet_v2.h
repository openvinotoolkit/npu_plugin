//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ie_core.hpp"
#include "model_helper.h"

class ModelMobileNet_V2_Helper : public ModelHelper {
public:
    explicit ModelMobileNet_V2_Helper();
};

//------------------------------------------------------------------------------
inline ModelMobileNet_V2_Helper::ModelMobileNet_V2_Helper() {
    _modelRelatedPath =
            "/KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8";
    loadModel();
}
