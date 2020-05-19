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

#include "helper_ie_core.h"
#include <gtest/gtest.h>
#include "models/precompiled_resnet.h"
#include "models/model_pooling.h"

//------------------------------------------------------------------------------
//      class Executable_Network_Parametric Params
//------------------------------------------------------------------------------
enum InferRequestFrom { ImportNetwork, LoadNetwork };

const static std::vector<InferRequestFrom> memoryOwners = {ImportNetwork, LoadNetwork};

//------------------------------------------------------------------------------
//      class Executable_Network_Parametric
//------------------------------------------------------------------------------
class Executable_Network_Parametric : public IE_Core_Helper,
                                      public ::testing::Test,
                                      public ::testing::WithParamInterface<InferRequestFrom> {
public:
    void SetUp() override;
    InferenceEngine::ExecutableNetwork executableNetwork;

    struct PrintToStringParamName {
        std::string operator()(testing::TestParamInfo<InferRequestFrom> const& info) const;
    };

protected:
    // ImportNetwork
    modelBlobInfo _modelBlobInfo = PrecompiledResNet_Helper::resnet50_dpu;

    // LoadNetwork
    InferenceEngine::CNNNetwork _cnnNetwork;
    ModelPooling_Helper _modelPoolingHelper;
};

inline void Executable_Network_Parametric::SetUp() {
    auto createFrom = GetParam();
    if (createFrom == ImportNetwork) {
        executableNetwork = ie.ImportNetwork(_modelBlobInfo.graphPath, pluginName);
    } else {
        _cnnNetwork = _modelPoolingHelper.network;
        executableNetwork = ie.LoadNetwork(_cnnNetwork, pluginName);
    }
}

inline std::string Executable_Network_Parametric::PrintToStringParamName::operator()(
        const testing::TestParamInfo<InferRequestFrom>& info) const {
    InferRequestFrom inferRequestFrom = info.param;
    if (inferRequestFrom == ImportNetwork) {
        return "ImportNetwork";
    } else if (inferRequestFrom == LoadNetwork) {
        return "LoadNetwork";
    } else {
        return "Unknown params";
    }
}
