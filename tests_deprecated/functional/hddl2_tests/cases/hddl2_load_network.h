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

#include <models/model_squeezenet_v1_1.h>
#include "core_api.h"
#include "gtest/gtest.h"

class LoadNetwork_Tests : public CoreAPI_Tests {
public:
    LoadNetwork_Tests();
    ModelSqueezenetV1_1_Helper modelHelper;
};

inline LoadNetwork_Tests::LoadNetwork_Tests() {
    network = modelHelper.getNetwork();
}

//------------------------------------------------------------------------------
class ExecutableNetwork_Tests : public LoadNetwork_Tests {
public:
    void SetUp() override;

protected:
    static InferenceEngine::ExecutableNetwork::Ptr _cacheExecNetwork;
};


