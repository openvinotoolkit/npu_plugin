// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,                 // Input Shapes
        std::vector<size_t>,                 // Kernel Shape
        size_t                               // Stride
> convParams;

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        convParams,                          // Convolution Params
        size_t                              // Output Channels
> multiOutputTestParams;

class MultioutputTest : public testing::WithParamInterface<multiOutputTestParams>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<multiOutputTestParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions


// namespace SubgraphTestsDefinitions {

// typedef std::tuple<
//         std::vector<size_t>,                 // Input Shapes
//         std::vector<size_t>,                 // Kernel Shape
//         size_t                               // Stride
// > convParams;

// typedef std::tuple<
//         ngraph::helpers::PoolingTypes,  // Pooling type, max or avg
//         std::vector<size_t>,            // Kernel size
//         std::vector<size_t>,            // Stride
//         std::vector<size_t>,            // Pad begin
//         std::vector<size_t>,            // Pad end
//         ngraph::op::RoundingType,       // Rounding type
//         ngraph::op::PadType,            // Pad type
//         bool                            // Exclude pad
// > poolParams;

// typedef std::tuple<
//         InferenceEngine::Precision,          // Network Precision
//         std::string,                         // Target Device
//         std::map<std::string, std::string>,  // Configuration
//         convParams,                          // Convolution Params
//         poolParams,                          // Pool Params
//         size_t                               // Output Channels
// > multiOutputTestParams;

// class MultioutputTest : public testing::WithParamInterface<multiOutputTestParams>,
//                      public LayerTestsUtils::LayerTestsCommon {
// public:
//     static std::string getTestCaseName(testing::TestParamInfo<multiOutputTestParams> obj);
//     InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

// protected:
//     void SetUp() override;
// };

// }  // namespace SubgraphTestsDefinitions
