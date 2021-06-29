// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"
#include "kmb_layer_test.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>   // Configuration
> mobilenetV2SlicedParameters;


class mobilenetV2SlicedTest :  public testing::WithParamInterface<mobilenetV2SlicedParameters>,
                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mobilenetV2SlicedParameters>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
