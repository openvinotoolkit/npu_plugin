// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"
#include "kmb_layer_test.hpp"

namespace SubgraphTestsDefinitions {

// ! [test_convolution:definition]
typedef std::tuple<
        LayerTestsDefinitions::activationParams,
        InferenceEngine::SizeVector,    // Kernel size
        InferenceEngine::SizeVector,    // Strides
        std::vector<ptrdiff_t>,         // Pad begin
        std::vector<ptrdiff_t>,         // Pad end
        InferenceEngine::SizeVector,    // Dilation
        size_t,                         // Num out channels
        ngraph::op::PadType             // Padding type
> convActTestParamsSet;


class ConvActTest :  public testing::WithParamInterface<convActTestParamsSet>,
                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convActTestParamsSet>& obj);

protected:
    void SetUp() override;

    void buildFloatFunction();

    void buildFQFunction();
};
// ! [test_convolution:definition]

}  // namespace SubgraphTestsDefinitions
