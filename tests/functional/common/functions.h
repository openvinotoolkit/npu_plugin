// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>

// create dummy network for tests
InferenceEngine::CNNNetwork buildSingleLayerSoftMaxNetwork();

// class encupsulated VPUXPlatform getting from environmental varriable
class PlatformEnvironment {

public:
    static const std::string PLATFORM;
};
