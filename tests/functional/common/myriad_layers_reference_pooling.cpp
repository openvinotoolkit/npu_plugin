// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "../common_single_layer_tests/pool_ref.hpp"
#include "../common_single_layer_tests/conv_ref.hpp"

using namespace InferenceEngine;

void ref_pooling_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct &params) {
    common_ref_pool_wrap<ie_fp16>({ src }, dst, params);
}
