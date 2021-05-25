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

class ModelSqueezenetV1_1_Helper : public ModelHelper {
public:
    explicit ModelSqueezenetV1_1_Helper();
};

//------------------------------------------------------------------------------
inline ModelSqueezenetV1_1_Helper::ModelSqueezenetV1_1_Helper() {
    _modelRelatedPath = "KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8";
    loadModel();
}
