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



