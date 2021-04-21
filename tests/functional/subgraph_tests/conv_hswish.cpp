// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbConvHSwishTest : public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeValidate() override {
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            throw LayerTestsUtils::KmbSkipTestException("Interpreter backend doesn't implement evaluate"
                                                        " method for OP HSwish  comparison fails");
        }
    }
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        constexpr int inChan = 16;
        constexpr int inWidth = 18;
        constexpr int inHeight = 18;
        constexpr int filtWidth = 5;
        constexpr int filtHeight = 5;

        const InferenceEngine::SizeVector inputShape = {1, inChan, inHeight, inWidth};

        const auto params = ngraph::builder::makeParams(ngraph::element::u8, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<uint8_t> conv_weights(inChan * filtHeight * filtWidth, 0);
        auto conv_weights_node = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::u8, ngraph::Shape{inChan, 1, 1, filtHeight, filtWidth}, conv_weights.data());

        auto conv2d_node = std::make_shared<ngraph::op::v1::GroupConvolution>(
                paramOuts.at(0), conv_weights_node->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        std::vector<uint8_t> bias_weights(inChan, 0);
        auto bias_weights_node = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::u8, ngraph::Shape{1, inChan, 1, 1}, bias_weights.data());
        auto bias_node = std::make_shared<ngraph::op::v1::Add>(conv2d_node->output(0), bias_weights_node->output(0));

        auto hswish_node = std::make_shared<ngraph::op::v4::HSwish>(bias_node->output(0));
        auto pool_node = std::make_shared<ngraph::op::v1::AvgPool>(hswish_node->output(0), ngraph::Strides{1, 1},
                                                                   ngraph::Shape{0, 0}, ngraph::Shape{0, 0},
                                                                   ngraph::Shape{14, 14}, true);

        auto mul_node = std::make_shared<ngraph::op::v1::Multiply>(hswish_node->output(0), pool_node->output(0));

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mul_node)};

        function = std::make_shared<ngraph::Function>(results, params, "KmbConvHSwishTest");

        threshold = 0.5f;
    }
};

TEST_F(KmbConvHSwishTest, CompareWithRefs) {
    Run();
}
}  // namespace
