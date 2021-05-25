//
// Copyright 2019-2020 Intel Corporation.
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
