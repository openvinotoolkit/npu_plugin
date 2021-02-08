//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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



