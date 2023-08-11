//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/functions.h"
#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXDeconvReluTest_VPU3700 : public LayerTestsUtils::KmbLayerTestsCommon {
    // [Track number: E#26428]
    void SkipBeforeLoad() override {
        if (getBackendName(*getCore()) == "VPUAL") {
            throw LayerTestsUtils::KmbSkipTestException("LoadNetwork throws an exception");
        }
    }
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }

    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        constexpr int inChan = 3;
        constexpr int outChan = 16;
        constexpr int inWidth = 14;
        constexpr int inHeight = 8;
        constexpr int filtWidth = 8;
        constexpr int filtHeight = 8;

        const InferenceEngine::SizeVector inputShape = {1, inChan, inHeight, inWidth};
        const InferenceEngine::SizeVector weightsShape{inChan, outChan, filtHeight, filtWidth};

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto weights = ngraph::builder::makeConstant<float>(ngraph::element::f16, weightsShape, {-1.0f}, false);

        const ngraph::Strides strides = {4, 4};
        const ngraph::CoordinateDiff pads_begin = {2, 2};
        const ngraph::CoordinateDiff pads_end = {2, 2};
        const ngraph::CoordinateDiff output_padding = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto auto_pad = ngraph::op::PadType::EXPLICIT;
        auto deconv2d_node = std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(
                params.at(0), weights, strides, pads_begin, pads_end, dilations, auto_pad, output_padding);

        auto relu_node = std::make_shared<ngraph::op::v0::Relu>(deconv2d_node->output(0));

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu_node)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXDeconvReluTest");

        threshold = 0.5f;
    }
};

TEST_F(VPUXDeconvReluTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace
