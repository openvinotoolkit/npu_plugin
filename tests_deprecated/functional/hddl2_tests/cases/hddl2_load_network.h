//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "core_api.h"
#include "gtest/gtest.h"
#include "models/models_constant.h"
#include "executable_network_factory.h"

class LoadNetwork_Tests : public CoreAPI_Tests {
public:
    LoadNetwork_Tests();
    const Models::ModelDesc modelToUse = Models::squeezenet1_1;
};

inline LoadNetwork_Tests::LoadNetwork_Tests() {
    network = ExecutableNetworkFactory::createCNNNetwork(modelToUse.pathToModel);
}

//------------------------------------------------------------------------------
class ExecutableNetwork_Tests : public LoadNetwork_Tests {
public:
    void SetUp() override;
};

inline void ExecutableNetwork_Tests::SetUp() {
    executableNetworkPtr = std::make_shared<InferenceEngine::ExecutableNetwork>(
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel));
}



