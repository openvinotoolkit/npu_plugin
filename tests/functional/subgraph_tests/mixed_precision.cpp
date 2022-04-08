// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "subgraph_tests/nce_tasks.hpp"

#include <ngraph_functions/builders.hpp>

namespace {

ov::Output<ov::Node> quantizeOutput(const ov::Output<ov::Node>& producer, const bool needQuant, const bool isMaxPool) {
    if (!needQuant) {
        // Bypass the quantization
        return producer;
    }

    const auto quantNCEOutRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    // MaxPool output range must match its input range
    const auto quantMaxPoolOutRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantRange = isMaxPool ? quantMaxPoolOutRange : quantNCEOutRange;
    const auto quantOp = NCETasksHelpers::quantize(producer, quantRange);

    return quantOp->output(0);
}

enum MixedMode {
    FP16toU8,
    U8toFP16,
};

typedef std::tuple<MixedMode, NCETasksHelpers::NCEOpType> MixedPrecisionParams;

class VPU3720NCEMixedPrecisionTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<MixedPrecisionParams> {
    void GenerateInputs() override {
        inputs.clear();
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (size_t i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
            InferenceEngine::InputInfo::CPtr info = infoIt->second;
            const uint32_t range = 10;
            const int32_t start_from = 0;
            InferenceEngine::Blob::Ptr blob =
                    FuncTestUtils::createAndFillBlob(info->getTensorDesc(), range, start_from);
            inputs.push_back(blob);
        }
    }
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const auto mixedMode = std::get<0>(GetParam());
        const bool isOutputQuantized = (mixedMode == FP16toU8);
        const auto nceOpType = std::get<1>(GetParam());
        const InferenceEngine::SizeVector inputShape = {1, 16, 32, 64};

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto nce_task = NCETasksHelpers::buildNCETask(paramOuts.at(0), nceOpType);

        const bool isMaxPool = (NCETasksHelpers::NCEOpType::MaxPooling == nceOpType);
        const auto maybeQuantOutput = quantizeOutput(nce_task->output(0), isOutputQuantized, isMaxPool);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(maybeQuantOutput)};

        function = std::make_shared<ngraph::Function>(results, params, "VPU3720NCEMixedPrecisionTest");

        threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MixedPrecisionParams>& paramInfo) {
        const auto mixedMode = std::get<0>(paramInfo.param);
        const auto isInputQuantized = (mixedMode == U8toFP16);
        const auto isOutputQuantized = (mixedMode == FP16toU8);
        const auto nceOpType = std::get<1>(paramInfo.param);

        const std::string inPrecision = isInputQuantized ? "inPRC=U8" : "inPRC=FP16";
        const std::string outPrecision = isOutputQuantized ? "outPRC=U8" : "outPRC=FP16";
        const auto opType = NCETasksHelpers::NCEOpTypeToString(nceOpType);

        return inPrecision + "_" + outPrecision + "_" + opType;
    }
};

TEST_P(VPU3720NCEMixedPrecisionTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<MixedMode> mixedMode = {FP16toU8, U8toFP16};
const std::vector<NCETasksHelpers::NCEOpType> nceOpType = {
        NCETasksHelpers::NCEOpType::AveragePooling, NCETasksHelpers::NCEOpType::Conv2d,
        NCETasksHelpers::NCEOpType::EltwiseAdd, NCETasksHelpers::NCEOpType::GroupConv2d,
        NCETasksHelpers::NCEOpType::MaxPooling};

INSTANTIATE_TEST_SUITE_P(conv2d_with_act, VPU3720NCEMixedPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(mixedMode), ::testing::ValuesIn(nceOpType)),
                         VPU3720NCEMixedPrecisionTest::getTestCaseName);

}  // namespace
