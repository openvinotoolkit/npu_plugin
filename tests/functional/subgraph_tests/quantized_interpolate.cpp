//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <common/utils.hpp>
#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include "vpux_private_config.hpp"

namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (dequantize)
//          |
//        (interp)
//          |
//       [output]
//          |
//      (quantize)
//

using QuantizedSEInterpTestParams = std::tuple<InferenceEngine::Precision,   // inPrc
                                               InferenceEngine::Precision,   // outPrc
                                               std::vector<float>,           // fqRanges
                                               InferenceEngine::SizeVector,  // inputShape
                                               std::vector<float>,           // scales
                                               LayerTestsUtils::TargetDevice>;
class VPUXQuantizedSEInterpSubGraphTest :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<QuantizedSEInterpTestParams> {
    void SetUp() override {
        std::vector<float> dataFQRanges;
        InferenceEngine::SizeVector inputShape;
        std::vector<float> interpScales;
        std::tie(inPrc, outPrc, dataFQRanges, inputShape, interpScales, targetDevice) = GetParam();
        threshold = 0.1f;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                              dataInLow, dataInHigh, dataOutLow, dataOutHigh);

        auto default_out_shape_node =
                ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {0, 0, 0, 0});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 2, 3});
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{4}, interpScales);

        auto interpolate4_attr = ngraph::opset4::Interpolate::InterpolateAttrs(
                ngraph::opset4::Interpolate::InterpolateMode::NEAREST,
                ngraph::opset4::Interpolate::ShapeCalcMode::SCALES, std::vector<size_t>{0, 0, 0, 0},
                std::vector<size_t>{0, 0, 0, 0}, ngraph::opset4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                ngraph::opset4::Interpolate::NearestMode::FLOOR, false, -0.75);

        auto interp = std::make_shared<ngraph::opset4::Interpolate>(dataFq, default_out_shape_node, scales_node,
                                                                    axes_node, interpolate4_attr);

        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ngraph::builder::makeFakeQuantize(interp, ngraph::element::f32, dataLevels, {}, outDataLow,
                                                             outDataHigh, outDataLow, outDataHigh);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outFq)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXQuantizedInterp");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedSEInterpTestParams> obj) {
        InferenceEngine::Precision ip;
        InferenceEngine::Precision op;
        std::vector<float> fqRanges;
        InferenceEngine::SizeVector inputShape;
        std::vector<float> interpScales;
        std::string targetDevice;
        std::tie(ip, op, fqRanges, inputShape, interpScales, targetDevice) = obj.param;

        std::ostringstream result;
        result << "InputPrec=" << ip.name() << "_";
        result << "OutputPrec=" << op.name() << "_";
        result << "FQ=" << vectorToString(fqRanges) << "_";
        result << "InputShape=" << vectorToString(inputShape) << "_";
        result << "InterpScales=" << vectorToString(interpScales) << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }
};

class VPUXQuantizedSEInterpSubGraphTest_VPU3720 : public VPUXQuantizedSEInterpSubGraphTest {
    void ConfigureNetwork() override {
        configuration[VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS)] = "enable-se-ptrs-operations=true";
    }
};

TEST_P(VPUXQuantizedSEInterpSubGraphTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXQuantizedSEInterpSubGraphTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

std::vector<std::vector<float>> fqRanges = {{0.0f, 255.0f, 0.0f, 255.0f}};

std::vector<InferenceEngine::SizeVector> inputShapes = {{1, 16, 16, 16}, {1, 48, 40, 40}};

std::vector<std::vector<float>> interpScales = {{1.0f, 1.0f, 2.0f, 2.0f}, {1.0f, 1.0f, 3.0f, 3.0f}};

const std::vector<InferenceEngine::Precision> netInPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::Precision> netOutputPrecisions = {InferenceEngine::Precision::FP16};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(netInPrecisions),
                                           ::testing::ValuesIn(netOutputPrecisions), ::testing::ValuesIn(fqRanges),
                                           ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(interpScales),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedInterp, VPUXQuantizedSEInterpSubGraphTest_VPU3720, basicCases,
                         VPUXQuantizedSEInterpSubGraphTest::getTestCaseName);

}  // namespace
