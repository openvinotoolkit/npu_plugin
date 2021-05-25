//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
    _modelRelatedPath = "/KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8";
    loadModel();
}
