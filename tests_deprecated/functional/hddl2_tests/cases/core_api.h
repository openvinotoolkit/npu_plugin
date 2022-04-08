//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <Inference.h>
#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <test_model_path.hpp>

#include "helper_ie_core.h"

class CoreAPI_Tests : public ::testing::Test,
                      public IE_Core_Helper {
public:
    InferenceEngine::CNNNetwork network;
    std::shared_ptr<InferenceEngine::ExecutableNetwork> executableNetworkPtr = nullptr;
    InferenceEngine::InferRequest inferRequest;
};
