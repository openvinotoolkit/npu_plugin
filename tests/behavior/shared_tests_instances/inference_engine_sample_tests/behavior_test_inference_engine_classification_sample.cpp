// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_inference_engine_classification_sample.hpp"

#define CS_KMB CSTestParams("kmbPlugin", model_path_fp16, "KMB")

CSTestParams correctPlugins[] = {
        CS_KMB,
};

INSTANTIATE_TEST_CASE_P(
        BehaviorTest,
        BehaviorTestInferenceEngineClassificationSample,
        ValuesIn(correctPlugins),
        getPluginName);
