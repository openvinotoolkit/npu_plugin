// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "../common_single_layer_tests/conv_ref.hpp"

using namespace InferenceEngine;


void ref_convolution_wrap(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          const uint16_t* weights,
                          size_t weights_size,
                          size_t bias_size,
                          const ParamsStruct& params) {
    uint16_t* bias_data = nullptr;
    if (bias_size) {
        bias_data = (uint16_t*) &weights[weights_size];
    }
    common_ref_convolution_wrap<ie_fp16>({ src }, dst, (const ie_fp16*)weights, weights_size, (ie_fp16*)bias_data, bias_size, params);
}

void ref_convolution(const Blob::Ptr src,
                     Blob::Ptr dst,
                     const ie_fp16* weights_data,
                     const ie_fp16* bias_data,
                     param_size kernel,
                     param_size stride,
                     param_size pad,
                     size_t group,
                     param_size dilation) {
    conv_common_params params;
    params.kernel.insert(X_AXIS, kernel.x);
    params.kernel.insert(Y_AXIS, kernel.y);
    params.stride.insert(X_AXIS, stride.x);
    params.stride.insert(Y_AXIS, stride.y);
    params.pads_begin.insert(X_AXIS, pad.x);
    params.pads_begin.insert(Y_AXIS, pad.y);
    params.dilation.insert(X_AXIS, dilation.x);
    params.dilation.insert(Y_AXIS, dilation.y);
    params.group = group;
    ref_conv_common<ie_fp16>({ src }, *dst.get(), weights_data, 0, bias_data, 0, params);
}
