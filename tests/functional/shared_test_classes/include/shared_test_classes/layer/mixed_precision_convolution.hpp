//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<InferenceEngine::SizeVector,              // kernel
                   InferenceEngine::SizeVector,              // stride
                   std::vector<ptrdiff_t>,                   // pad begin
                   std::vector<ptrdiff_t>,                   // pad end
                   InferenceEngine::SizeVector,              // dilation
                   size_t,                                   // output channels
                   size_t,                                   // quant levels (to lower to I8 or I4)
                   ngraph::helpers::QuantizationGranularity  // quant granularity (per tensor/channel)
                   >
        mixedPrecisionConvSpecificParams;
typedef std::tuple<mixedPrecisionConvSpecificParams,  // specific params
                   InferenceEngine::Precision,        // network precision
                   InferenceEngine::SizeVector,       // input shape
                   ov::test::TargetDevice             // target device
                   >
        mixedPrecisionConvLayerTestParamsSet;

class MixedPrecisionConvLayerTest :
        public testing::WithParamInterface<mixedPrecisionConvLayerTestParamsSet>,
        virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mixedPrecisionConvLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
