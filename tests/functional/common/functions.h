// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

// create dummy network for tests
InferenceEngine::CNNNetwork buildSingleLayerSoftMaxNetwork();

std::string getBackendName(const InferenceEngine::Core& core);
std::vector<std::string> getAvailableDevices(const InferenceEngine::Core& core);

// class encupsulated VPUXPlatform getting from environmental variable
class PlatformEnvironment {
public:
    static const std::string PLATFORM;
};
