// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "../common_single_layer_tests/deconv_ref.hpp"

using namespace InferenceEngine;


void ref_deconvolution_wrap(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          const uint16_t* weights,
                          size_t weights_size,
                          size_t bias_size,
                          const ParamsStruct& params) {
    const ie_fp16* bias_data = nullptr;
    if (bias_size) {
        bias_data = reinterpret_cast<const ie_fp16*>(weights + weights_size);
    }
    common_ref_deconvolution_wrap<ie_fp16>({src}, dst, reinterpret_cast<const ie_fp16*>(weights), weights_size, bias_data, bias_size, params);
}
